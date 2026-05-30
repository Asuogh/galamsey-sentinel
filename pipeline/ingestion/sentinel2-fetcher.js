/**
 * @file sentinel2-fetcher.js
 * @description Fetches, filters, prepares, and exports Sentinel-2 L2A
 *              multispectral satellite imagery over target Galamsey zones
 *              in Ghana using the Google Earth Engine JavaScript API.
 *
 * REPORT CONTEXT — What is Sentinel-2 and Why These Bands?
 * ─────────────────────────────────────────────────────────────────────────────
 * Sentinel-2 is a pair of twin satellites (2A and 2B) operated by the European
 * Space Agency (ESA) as part of the Copernicus Earth Observation Programme.
 * They carry a MultiSpectral Instrument (MSI) that captures imagery across
 * 13 spectral bands at resolutions of 10m, 20m, and 60m.
 *
 * We use the Level-2A (L2A) product, which means the raw at-sensor radiance
 * values have already been corrected for atmospheric effects by ESA's Sen2Cor
 * processor, giving us Surface Reflectance — the actual reflectance of the
 * ground surface — essential for consistent cross-date comparisons.
 *
 * The four bands selected serve specific purposes in Galamsey detection:
 *
 *  ┌──────┬────────────────┬────────────┬──────────────────────────────────┐
 *  │ Band │ Name           │ Resolution │ Purpose in This Project          │
 *  ├──────┼────────────────┼────────────┼──────────────────────────────────┤
 *  │  B2  │ Blue (490nm)   │ 10m        │ Visual composite; water clarity  │
 *  │  B3  │ Green (560nm)  │ 10m        │ Visual composite; turbidity      │
 *  │  B4  │ Red (665nm)    │ 10m        │ Visual composite; soil/sediment  │
 *  │  B8  │ NIR (842nm)    │ 10m        │ Vegetation health; NDVI; NDWI   │
 *  └──────┴────────────────┴────────────┴──────────────────────────────────┘
 *
 * GEE Dataset ID: "COPERNICUS/S2_SR_HARMONIZED"
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * REPORT CONTEXT — Why Export to Earth Engine Assets?
 * ─────────────────────────────────────────────────────────────────────────────
 * GEE offers three export destinations for image data:
 *
 *  1. Google Drive    → Requires the authenticated identity (Service Account)
 *                       to have Drive access under a user quota. Service
 *                       Accounts do not have personal Drive storage, so this
 *                       fails unless Drive is explicitly shared with the account.
 *
 *  2. Cloud Storage   → Requires a Google Cloud project with an active billing
 *                       account. Not available on free/student projects.
 *
 *  3. Earth Engine Assets (toAsset) → Writes directly into GEE's own managed
 *                       storage under your registered Cloud Project. This is
 *                       entirely free within the GEE quota, requires NO billing
 *                       account, and is the most efficient destination for
 *                       images that will be used in subsequent GEE operations
 *                       (e.g., feeding into our CNN preprocessing pipeline).
 *                       The exported image becomes a first-class GEE asset
 *                       that can be loaded with ee.Image("asset/path") in any
 *                       future script with zero re-download cost.
 *
 * PREREQUISITE: The asset folder must exist before export. Create it once via:
 *   ee.data.createFolder("projects/galamsey-sentinel/assets/sentinel2_tiles")
 * or in the GEE Code Editor: Assets → New → Folder.
 * The createAssetFolderIfMissing() function in this file handles this
 * automatically at pipeline startup.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @module sentinel2-fetcher
 */

import ee from "@google/earthengine";
import path from "path";
import { fileURLToPath } from "url";
import "dotenv/config";
import { authenticateGEE, logger } from "./gee-auth.js";

// ─── ES Module __dirname Shim ─────────────────────────────────────────────────
const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// ─── GEE Collection + Band Constants ─────────────────────────────────────────

/** GEE Image Collection ID for Sentinel-2 Surface Reflectance (Harmonized). */
const S2_COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED";

/** The four spectral bands extracted per image patch. */
const TARGET_BANDS = ["B2", "B3", "B4", "B8"];

/**
 * Maximum acceptable cloud cover percentage per scene (0–100).
 * Scenes above this threshold are discarded before compositing.
 */
const MAX_CLOUD_PERCENTAGE = 20;

// ─── Tiling Constants ─────────────────────────────────────────────────────────

/**
 * Maximum tile area in km² before a bounding box is subdivided.
 * Kept conservatively below the GEE 2500 km² hard limit.
 */
const MAX_TILE_AREA_KM2 = 2000;

/** Milliseconds to wait between sequential tile pipeline runs. */
const INTER_TILE_DELAY_MS = 3000;

/** Maximum retry attempts per tile for transient GEE API errors. */
const MAX_RETRIES = 3;

/** Base delay in ms for exponential backoff. Attempt N waits N × base. */
const RETRY_BASE_DELAY_MS = 10_000;

// ─── Asset Export Constants ───────────────────────────────────────────────────

/**
 * Root Earth Engine Asset path for all exported tiles.
 *
 * REPORT CONTEXT — GEE Asset Path Structure:
 * ─────────────────────────────────────────────────────────────────────────────
 * Earth Engine asset paths follow the pattern:
 *   projects/<cloud-project-id>/assets/<folder>/<asset-name>
 *
 * The cloud project ID ("galamsey-sentinel") must match the Google Cloud
 * project registered with Earth Engine at signup.earthengine.google.com.
 * The folder ("sentinel2_tiles") must be pre-created in the EE asset registry.
 * Each exported tile is stored as a named image asset under this folder.
 *
 * Once exported, any asset in this path can be loaded back into GEE via:
 *   const img = ee.Image("projects/galamsey-sentinel/assets/sentinel2_tiles/tileName");
 * making it immediately available to the cloud-masking and patch-extraction
 * stages of our pipeline without any file download.
 * ─────────────────────────────────────────────────────────────────────────────
 */
const ASSET_ROOT = process.env.GEE_ASSET_ROOT ||
  "projects/galamsey-sentinel/assets/sentinel2_tiles";

/**
 * Export pixel size in metres. Matches Sentinel-2's native B2/B3/B4/B8
 * ground sampling distance so no resampling artefacts are introduced.
 */
const EXPORT_SCALE_METRES = 10;

/**
 * Coordinate Reference System for exported assets.
 * EPSG:4326 (WGS84) keeps all pipeline coordinates consistent so that
 * GeoTIFF coordinates, PostGIS coordinates, and Leaflet map coordinates
 * share the same reference frame with no reprojection required.
 */
const EXPORT_CRS = "EPSG:4326";

/**
 * Polling interval in ms between ee.data.getTaskStatus() calls.
 * 15 seconds is a reasonable balance between responsiveness and API load.
 */
const POLL_INTERVAL_MS = 15_000;

/**
 * Maximum time in ms to wait for a single export task before timing out.
 * Default 2 hours. Asset exports at 10m resolution can take 30–60 minutes
 * on a busy GEE server, but are generally faster than Drive exports because
 * GEE writes to its own internal storage.
 */
const EXPORT_TIMEOUT_MS =
  parseInt(process.env.GEE_EXPORT_TIMEOUT_MS) || 7_200_000;

// ─── Study Regions ────────────────────────────────────────────────────────────

/**
 * REPORT CONTEXT — Target Study Regions:
 * ─────────────────────────────────────────────────────────────────────────────
 * These bounding boxes cover the highest-risk Galamsey zones in Ghana's
 * Western and Ashanti Regions, specifically the watersheds of the Pra,
 * Ankobra, and Birim rivers. Coordinates are WGS84 [West, South, East, North].
 * Each region is automatically tiled if its area exceeds MAX_TILE_AREA_KM2.
 * ─────────────────────────────────────────────────────────────────────────────
 */
const STUDY_REGIONS = {
  pra_river_basin: {
    name: "Pra River Basin (Western/Ashanti Region)",
    bbox: [-2.3, 5.2, -1.5, 6.1],
  },
  ankobra_basin: {
    name: "Ankobra River Basin (Western Region)",
    bbox: [-2.6, 4.9, -1.9, 5.7],
  },
  birim_basin: {
    name: "Birem River Basin (Eastern Region)",
    bbox: [-1.2, 5.8, -0.4, 6.5],
  },
};

// ─── Utility: Sleep ───────────────────────────────────────────────────────────
/** @param {number} ms @returns {Promise<void>} */
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ─── Utility: Area Estimator ──────────────────────────────────────────────────
/**
 * Estimates bounding box area in km² via equirectangular approximation.
 * @param {number[]} bbox - [west, south, east, north]
 * @returns {number}
 */
function estimateAreaKm2(bbox) {
  const [west, south, east, north] = bbox;
  const midLat = (south + north) / 2;
  const latKm  = (north - south) * 111;
  const lonKm  = (east  - west)  * 111 * Math.cos((midLat * Math.PI) / 180);
  return Math.round(latKm * lonKm);
}

// ─── Utility: Sanitise Asset Name ─────────────────────────────────────────────
/**
 * Converts an arbitrary string into a valid GEE asset name segment.
 *
 * REPORT CONTEXT — GEE Asset Naming Rules:
 * ─────────────────────────────────────────────────────────────────────────────
 * Earth Engine asset names are more permissive than Drive filenames but still
 * reject spaces, parentheses, and most special characters. Only alphanumeric
 * characters, underscores, and hyphens are reliably safe. We normalise all
 * input to this character set and collapse repeated separators.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {string} name - Raw string (e.g. tileId or region name).
 * @returns {string} GEE-safe asset name segment.
 *
 * @example
 * sanitiseAssetName("Pra River Basin (Western) tile_r0_c1")
 * // → "Pra_River_Basin_Western_tile_r0_c1"
 */
function sanitiseAssetName(name) {
  return name
    .replace(/[^a-zA-Z0-9_-]/g, "_") // replace non-alphanumeric/underscore/hyphen
    .replace(/_+/g, "_")              // collapse consecutive underscores
    .replace(/^_|_$/g, "");           // trim leading/trailing underscores
}

// ─── Utility: BBox Validator ──────────────────────────────────────────────────
/**
 * Validates a bounding box for structure, logical ordering, and WGS84 range.
 * @param {number[]} bbox - [west, south, east, north]
 * @returns {{ isValid: boolean, reason?: string }}
 */
function validateBoundingBox(bbox) {
  if (!Array.isArray(bbox) || bbox.length !== 4) {
    return { isValid: false, reason: "BBox must be [W, S, E, N] — 4 numbers." };
  }
  const [west, south, east, north] = bbox;
  if (west  >= east)  return { isValid: false, reason: `West (${west}) must be < East (${east}).` };
  if (south >= north) return { isValid: false, reason: `South (${south}) must be < North (${north}).` };
  if (south < -90 || north > 90 || west < -180 || east > 180) {
    return { isValid: false, reason: "Coordinates outside valid WGS84 range." };
  }
  return { isValid: true };
}

// ─── Tiling Engine ────────────────────────────────────────────────────────────
/**
 * Subdivides a bounding box into a regular grid of tiles each ≤ MAX_TILE_AREA_KM2.
 *
 * REPORT CONTEXT — Tiling Algorithm:
 * ─────────────────────────────────────────────────────────────────────────────
 *  1. Estimate total area. If within limit, return a single tile unchanged.
 *  2. Compute minimum tiles needed: ceil(totalArea / MAX_TILE_AREA_KM2).
 *  3. Derive grid: rows = cols = ceil(sqrt(tilesNeeded)) for square cells.
 *  4. Emit each [W, S, E, N] cell with a unique tileId for traceability.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {number[]} bbox
 * @param {string}   regionName
 * @returns {Array<{ tileId: string, bbox: number[], row: number, col: number }>}
 */
function generateTiles(bbox, regionName) {
  const [west, south, east, north] = bbox;
  const totalArea = estimateAreaKm2(bbox);

  if (totalArea <= MAX_TILE_AREA_KM2) {
    logger.info(`[${regionName}] Area ~${totalArea} km² within limit. No tiling needed.`);
    return [{ tileId: `${regionName}_tile_r0_c0`, bbox, row: 0, col: 0 }];
  }

  const tilesNeeded = Math.ceil(totalArea / MAX_TILE_AREA_KM2);
  const gridSize    = Math.ceil(Math.sqrt(tilesNeeded));
  const rows        = gridSize;
  const cols        = gridSize;
  const tileWidth   = (east  - west)  / cols;
  const tileHeight  = (north - south) / rows;
  const estTileArea = estimateAreaKm2([west, south, west + tileWidth, south + tileHeight]);

  logger.info(
    `[${regionName}] Area ~${totalArea} km² exceeds limit. ` +
    `Auto-tiling into ${rows}×${cols} grid (~${estTileArea} km² each).`
  );

  const tiles = [];
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      tiles.push({
        tileId : `${regionName}_tile_r${row}_c${col}`,
        bbox   : [
          parseFloat((west  + col       * tileWidth ).toFixed(6)),
          parseFloat((south + row       * tileHeight).toFixed(6)),
          parseFloat((west  + (col + 1) * tileWidth ).toFixed(6)),
          parseFloat((south + (row + 1) * tileHeight).toFixed(6)),
        ],
        row,
        col,
      });
    }
  }
  return tiles;
}

// ─── Asset Folder Bootstrap ───────────────────────────────────────────────────
/**
 * Ensures the target Earth Engine asset folder exists before any export task
 * is submitted. Creates it if absent; silently continues if it already exists.
 *
 * REPORT CONTEXT — Why This Is Necessary:
 * ─────────────────────────────────────────────────────────────────────────────
 * Unlike Google Drive (which auto-creates folders), the Earth Engine asset
 * registry requires the destination folder to exist BEFORE a toAsset() export
 * task is started. Submitting an export to a non-existent asset path produces
 * a silent task failure with a cryptic "asset not found" error message that
 * only appears when the task is polled minutes later.
 *
 * By calling ee.data.createFolder() at startup and treating an
 * "already exists" error as a no-op, we make the pipeline fully idempotent —
 * safe to run multiple times without manual asset management.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {string} assetPath - Full EE asset folder path to verify/create.
 * @returns {Promise<void>}
 */
async function createAssetFolderIfMissing(assetPath) {
  logger.info(`[ASSET-SETUP] Verifying asset folder: ${assetPath}`);

  await new Promise((resolve, reject) => {
    // ee.data.createFolder() is the Node.js API method to create an asset
    // folder. The second argument (false) means do NOT create parent folders
    // recursively — our root project path must already exist.
    ee.data.createFolder(assetPath, false, (result, error) => {
      if (!error) {
        logger.info(`[ASSET-SETUP] ✓ Asset folder created: ${assetPath}`);
        return resolve();
      }

      // GEE returns a plain string error, not an Error object.
      // "Cannot overwrite" and "already exists" both indicate the folder
      // is present — this is the expected state on subsequent pipeline runs.
      const errorStr = String(error).toLowerCase();
      if (
        errorStr.includes("already exists") ||
        errorStr.includes("cannot overwrite")
      ) {
        logger.info(`[ASSET-SETUP] ✓ Asset folder already exists: ${assetPath}`);
        return resolve();
      }

      // Any other error (e.g. parent project path wrong, insufficient
      // permissions, or the EE project not registered) is genuinely fatal.
      reject(
        new Error(
          `[ASSET-SETUP] Failed to create asset folder "${assetPath}": ${error}. ` +
          `Ensure the Cloud Project ID in ASSET_ROOT matches the project ` +
          `registered at signup.earthengine.google.com and the service account ` +
          `has the "Earth Engine Resource Writer" IAM role.`
        )
      );
    });
  });
}

// ─── Asset Existence Check ────────────────────────────────────────────────────
/**
 * Checks whether an EE image asset already exists at the given path.
 * Used to skip re-export of tiles that completed in a previous pipeline run,
 * making the pipeline safely resumable after crashes or timeouts.
 *
 * @param {string} assetId - Full EE asset path to check.
 * @returns {Promise<boolean>} True if the asset exists, false otherwise.
 */
async function assetExists(assetId) {
  return new Promise((resolve) => {
    ee.data.getAsset(assetId, (result, error) => {
      // If no error and result is populated, the asset exists.
      resolve(!error && result != null);
    });
  });
}

// ─── Single-Tile Fetcher ──────────────────────────────────────────────────────
/**
 * Fetches a Sentinel-2 median composite for one validated tile bounding box.
 * Includes exponential backoff retry for transient GEE API errors.
 *
 * @param {object}   params
 * @param {number[]} params.bbox
 * @param {string}   params.startDate
 * @param {string}   params.endDate
 * @param {string}   params.tileId
 * @param {number}   [params.attempt=1]
 * @returns {Promise<{ image: ee.Image, region: ee.Geometry, metadata: object } | null>}
 */
async function fetchTile({ bbox, startDate, endDate, tileId, attempt = 1 }) {
  const tag = `[FETCH][${tileId}]`;

  try {
    const [west, south, east, north] = bbox;
    const region = ee.Geometry.Rectangle([west, south, east, north], null, false);

    const collection = ee
      .ImageCollection(S2_COLLECTION_ID)
      .filterBounds(region)
      .filterDate(startDate, endDate)
      .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PERCENTAGE))
      .select(TARGET_BANDS);

    const sceneCount = await new Promise((resolve, reject) => {
      collection.size().evaluate((result, error) => {
        if (error) reject(new Error(`GEE evaluate() failed: ${error}`));
        else resolve(result);
      });
    });

    if (sceneCount === 0) {
      logger.warn(`${tag} 0 scenes found after cloud filter. Skipping tile.`);
      return null;
    }

    logger.info(`${tag} ${sceneCount} scenes found. Building median composite...`);

    // REPORT CONTEXT — Median Composite:
    // The MEDIAN reducer picks the middle reflectance value per pixel across
    // all cloud-filtered scenes, naturally suppressing remaining cloud/shadow
    // outliers without requiring an explicit per-pixel mask at this stage.
    const medianComposite = collection.median().clip(region);

    const bandNames = await new Promise((resolve, reject) => {
      medianComposite.bandNames().evaluate((result, error) => {
        if (error) reject(new Error(`Band name retrieval failed: ${error}`));
        else resolve(result);
      });
    });

    const metadata = {
      tileId,
      bbox,
      startDate,
      endDate,
      sceneCount,
      bands: bandNames,
      cloudFilterPct : MAX_CLOUD_PERCENTAGE,
      compositeMethod: "median",
      collectionId   : S2_COLLECTION_ID,
      createdAt      : new Date().toISOString(),
    };

    logger.info(
      `${tag} Composite ready. Bands: [${bandNames.join(", ")}] | Scenes: ${sceneCount}`
    );

    return { image: medianComposite, region, metadata };

  } catch (err) {
    const isRetryable =
      err.message.toLowerCase().includes("quota")             ||
      err.message.toLowerCase().includes("too many requests") ||
      err.message.toLowerCase().includes("rate")              ||
      err.message.toLowerCase().includes("timeout")           ||
      err.message.toLowerCase().includes("network");

    if (isRetryable && attempt <= MAX_RETRIES) {
      const backoffMs = RETRY_BASE_DELAY_MS * attempt;
      logger.warn(
        `${tag} Transient error (attempt ${attempt}/${MAX_RETRIES}): ${err.message}. ` +
        `Retrying in ${backoffMs / 1000}s...`
      );
      await sleep(backoffMs);
      return fetchTile({ bbox, startDate, endDate, tileId, attempt: attempt + 1 });
    }

    throw new Error(`${tag} Failed after ${attempt - 1} retries: ${err.message}`);
  }
}

// ─── Export: Submit toAsset Task ──────────────────────────────────────────────
/**
 * Submits a GEE export task that writes a composite tile into Earth Engine
 * Assets using ee.batch.Export.image.toAsset().
 *
 * REPORT CONTEXT — ee.batch.Export.image.toAsset() Parameters:
 * ─────────────────────────────────────────────────────────────────────────────
 * Unlike toDrive() or toCloudStorage(), toAsset() writes the output image
 * directly into GEE's managed internal storage. Key parameters:
 *
 *  • image        : The ee.Image to export (our median composite).
 *  • description  : The human-readable task label in the GEE Task Manager.
 *                   Shown in the Code Editor UI — does NOT affect asset path.
 *  • assetId      : The FULL asset path including folder and file name.
 *                   Must be unique; exporting to an existing asset ID fails
 *                   unless the asset is deleted first.
 *  • scale        : Output pixel size in metres (10m = native S2 resolution).
 *  • region       : ee.Geometry defining the export boundary. Only pixels
 *                   inside this geometry are written to the asset.
 *  • crs          : Output coordinate reference system (EPSG:4326 / WGS84).
 *  • maxPixels    : Safety ceiling. We raise to 1e9 to handle large 10m tiles
 *                   without triggering GEE's default 1e8 pixel limit error.
 *  • pyramidingPolicy : Controls how the asset is downsampled for zoom levels.
 *                   "MEAN" is appropriate for continuous reflectance values —
 *                   it averages adjacent pixels rather than picking the nearest
 *                   or maximum, preserving radiometric accuracy at lower zooms.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {object}      params
 * @param {ee.Image}    params.image     - Composited image to export.
 * @param {ee.Geometry} params.region    - Tile geometry for the export boundary.
 * @param {object}      params.metadata  - Tile metadata from fetchTile().
 * @param {string}      [params.assetRoot] - Override for the root asset path.
 *
 * @returns {Promise<{ taskId: string, assetId: string, status: "SUBMITTED" }>}
 * @throws {Error} If task.start() fails (bad asset path, auth expired, etc.).
 */
async function submitExportTask({ image, region, metadata, assetRoot = ASSET_ROOT }) {
  const tag = `[EXPORT][${metadata.tileId}]`;

  // Build a GEE-safe asset name and construct the full asset path.
  const year      = new Date(metadata.startDate).getFullYear();
  const assetName = sanitiseAssetName(`${metadata.tileId}_${year}`);
  const assetId   = `${assetRoot}/${assetName}`;

  logger.info(`${tag} Submitting toAsset export task.`);
  logger.info(`${tag} Asset destination: ${assetId}`);
  logger.info(
    `${tag} Scale: ${EXPORT_SCALE_METRES}m | CRS: ${EXPORT_CRS} | ` +
    `Bands: [${metadata.bands.join(", ")}]`
  );

  // Build the export task configuration. No API call yet — this is local.
  const task = ee.batch.Export.image.toAsset({
    image,
    description     : assetName,   // Task label in GEE Code Editor Task Manager
    assetId         : assetId,     // Full EE asset path (folder + name)
    scale           : EXPORT_SCALE_METRES,
    region          : region,      // Clip to exact tile boundary
    crs             : EXPORT_CRS,
    maxPixels       : 1e9,         // Raise ceiling for large 10m tiles
    pyramidingPolicy: {            // Per-band downsampling strategy
      B2: "MEAN",
      B3: "MEAN",
      B4: "MEAN",
      B8: "MEAN",
    },
  });

  // task.start() registers the task in GEE's queue. This is the only
  // network-touching call in this function. We wrap it in a Promise
  // to surface errors as rejections rather than thrown exceptions.
  await new Promise((resolve, reject) => {
    try {
      task.start();
      resolve();
    } catch (startError) {
      reject(
        new Error(
          `${tag} task.start() failed: ${startError.message}. ` +
          `Common causes: ` +
          `(1) Asset folder does not exist — run createAssetFolderIfMissing(). ` +
          `(2) Asset ID already exists — delete the old asset or use a new name. ` +
          `(3) GEE auth token expired — re-run authenticateGEE(). ` +
          `(4) Cloud Project ID in assetId does not match your registered project.`
        )
      );
    }
  });

  const taskId = task.id ?? "unknown";

  logger.info(
    `${tag} Task submitted. GEE Task ID: ${taskId}. ` +
    `Monitor at: https://code.earthengine.google.com/tasks`
  );

  return { taskId, assetId, status: "SUBMITTED" };
}

// ─── Export: Poll Task Status ─────────────────────────────────────────────────
/**
 * Polls a GEE export task until it reaches a terminal state or times out.
 *
 * REPORT CONTEXT — GEE Task States:
 * ─────────────────────────────────────────────────────────────────────────────
 *   READY      → Queued, waiting for a GEE worker.
 *   RUNNING    → Actively rendering and writing the asset.
 *   COMPLETED  → Asset successfully written to EE asset registry.
 *   FAILED     → Rendering failed (check error_message for details).
 *   CANCELLED  → Manually cancelled via Task Manager UI or API.
 *
 * toAsset() exports are typically faster than toDrive() because GEE writes
 * to its own internal storage (no OAuth transfer to an external service).
 * Expect 10–30 minutes per tile at 10m resolution on the free tier.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {string} taskId  - GEE task ID from submitExportTask().
 * @param {string} assetId - Full asset path, used in log messages.
 * @returns {Promise<{
 *   taskId: string,
 *   state: "COMPLETED"|"FAILED"|"CANCELLED"|"TIMEOUT",
 *   assetId: string,
 *   elapsedMs: number,
 *   errorMessage?: string
 * }>}
 */
async function pollExportTask(taskId, assetId) {
  const tag = `[POLL][${assetId.split("/").pop()}]`;
  const startTime = Date.now();

  logger.info(
    `${tag} Polling every ${POLL_INTERVAL_MS / 1000}s. ` +
    `Timeout: ${EXPORT_TIMEOUT_MS / 60_000} min.`
  );

  const TERMINAL_STATES = new Set(["COMPLETED", "FAILED", "CANCELLED"]);

  while (true) {
    const elapsedMs = Date.now() - startTime;

    if (elapsedMs > EXPORT_TIMEOUT_MS) {
      logger.warn(
        `${tag} Timed out after ${Math.round(elapsedMs / 60_000)} min. ` +
        `Task ${taskId} may still be running. ` +
        `Check: https://code.earthengine.google.com/tasks`
      );
      return { taskId, state: "TIMEOUT", assetId, elapsedMs };
    }

    let statusList;
    try {
      statusList = await new Promise((resolve, reject) => {
        ee.data.getTaskStatus([taskId], (result, error) => {
          if (error) reject(new Error(`getTaskStatus failed: ${error}`));
          else resolve(result);
        });
      });
    } catch (pollError) {
      // Intermittent API errors during polling are not fatal — log and retry.
      logger.warn(`${tag} Status check failed: ${pollError.message}. Retrying...`);
      await sleep(POLL_INTERVAL_MS);
      continue;
    }

    const taskStatus = statusList?.[0];
    const state      = taskStatus?.state ?? "UNKNOWN";

    const mm = String(Math.floor(elapsedMs / 60_000)).padStart(2, "0");
    const ss = String(Math.floor((elapsedMs % 60_000) / 1000)).padStart(2, "0");

    logger.info(`${tag} State: ${state} | Elapsed: ${mm}:${ss} | Task: ${taskId}`);

    if (TERMINAL_STATES.has(state)) {
      if (state === "COMPLETED") {
        logger.info(
          `${tag} ✓ Export COMPLETED in ${mm}m ${ss}s. Asset: ${assetId}`
        );
      } else {
        const errorMessage =
          taskStatus?.error_message ?? taskStatus?.description ?? "No details.";
        logger.error(`${tag} ✗ Export ${state}: ${errorMessage}`);
        return { taskId, state, assetId, elapsedMs, errorMessage };
      }
      return { taskId, state, assetId, elapsedMs };
    }

    await sleep(POLL_INTERVAL_MS);
  }
}

// ─── Region Fetcher + Exporter ────────────────────────────────────────────────
/**
 * Fetches Sentinel-2 composites for an entire region and exports every
 * successful tile to Earth Engine Assets.
 *
 * Per-tile pipeline:
 *   1. Check if asset already exists → skip if so (resumable pipeline).
 *   2. Fetch median composite        → fetchTile().
 *   3. Submit toAsset export task    → submitExportTask().
 *   4. Poll until terminal state     → pollExportTask().
 *   5. Wait INTER_TILE_DELAY_MS before next tile.
 *
 * @param {object}   params
 * @param {number[]} params.bbox
 * @param {string}   params.startDate
 * @param {string}   params.endDate
 * @param {string}   [params.regionName]
 *
 * @returns {Promise<{
 *   regionName   : string,
 *   tileCount    : number,
 *   successCount : number,
 *   skippedCount : number,
 *   failedCount  : number,
 *   exportResults: object[],
 *   errors       : Array<{ tileId: string, error: string }>
 * }>}
 */
async function fetchSentinel2Composite({
  bbox,
  startDate,
  endDate,
  regionName = "unnamed-region",
}) {
  logger.info(`\n${"─".repeat(60)}`);
  logger.info(`[${regionName}] Starting region fetch + asset export.`);
  logger.info(
    `[${regionName}] Date: ${startDate} → ${endDate} | ` +
    `Cloud: ${MAX_CLOUD_PERCENTAGE}% | Bands: ${TARGET_BANDS.join(", ")}`
  );
  logger.info(`[${regionName}] Asset root: ${ASSET_ROOT}`);

  const validation = validateBoundingBox(bbox);
  if (!validation.isValid) {
    throw new Error(`[${regionName}] Invalid bounding box: ${validation.reason}`);
  }

  const tiles = generateTiles(bbox, regionName);
  logger.info(`[${regionName}] Processing ${tiles.length} tile(s) sequentially...`);

  const exportResults = [];
  const errors        = [];
  let   skippedCount  = 0;

  for (let i = 0; i < tiles.length; i++) {
    const { tileId, bbox: tileBbox, row, col } = tiles[i];

    logger.info(
      `\n[${regionName}] ── Tile ${i + 1}/${tiles.length} ──────────────────────`
    );
    logger.info(
      `[${regionName}] ID: ${tileId} (row ${row}, col ${col}) | ` +
      `bbox: [${tileBbox.map((v) => v.toFixed(4)).join(", ")}]`
    );

    try {
      // ── Step 1: Skip if asset already exists ─────────────────────────────
      // This makes the pipeline fully resumable. If a previous run exported
      // this tile successfully, we skip it without re-submitting a task.
      const year         = new Date(startDate).getFullYear();
      const assetName    = sanitiseAssetName(`${tileId}_${year}`);
      const expectedAssetId = `${ASSET_ROOT}/${assetName}`;

      const alreadyExported = await assetExists(expectedAssetId);
      if (alreadyExported) {
        logger.info(
          `[${tileId}] Asset already exists at ${expectedAssetId}. Skipping.`
        );
        skippedCount++;
        if (i < tiles.length - 1) await sleep(INTER_TILE_DELAY_MS);
        continue;
      }

      // ── Step 2: Fetch median composite ───────────────────────────────────
      const fetchResult = await fetchTile({ bbox: tileBbox, startDate, endDate, tileId });

      if (fetchResult === null) {
        skippedCount++;
        logger.warn(`[${tileId}] Skipped — no cloud-free imagery.`);
        if (i < tiles.length - 1) await sleep(INTER_TILE_DELAY_MS);
        continue;
      }

      const { image, region, metadata } = fetchResult;

      // ── Step 3: Submit toAsset export task ───────────────────────────────
      const submission = await submitExportTask({ image, region, metadata });

      // ── Step 4: Poll until the export task reaches a terminal state ───────
      const pollResult = await pollExportTask(submission.taskId, submission.assetId);

      exportResults.push({
        tileId,
        ...submission,
        ...pollResult,
        sceneCount: metadata.sceneCount,
      });

      if (pollResult.state !== "COMPLETED") {
        errors.push({
          tileId,
          error: `Export ${pollResult.state}: ${pollResult.errorMessage ?? "timeout"}`,
        });
      }

    } catch (tileError) {
      errors.push({ tileId, error: tileError.message });
      logger.error(`[${tileId}] Pipeline error: ${tileError.message}`);
    }

    if (i < tiles.length - 1) {
      logger.info(`[${regionName}] Waiting ${INTER_TILE_DELAY_MS / 1000}s...`);
      await sleep(INTER_TILE_DELAY_MS);
    }
  }

  const completedCount = exportResults.filter((r) => r.state === "COMPLETED").length;

  logger.info(`\n[${regionName}] Region complete.`);
  logger.info(
    `[${regionName}] ✓ Exported: ${completedCount} | ` +
    `⚠ Skipped: ${skippedCount} | ✗ Failed: ${errors.length}`
  );

  return {
    regionName,
    tileCount    : tiles.length,
    successCount : completedCount,
    skippedCount,
    failedCount  : errors.length,
    exportResults,
    errors,
  };
}

// ─── Main Orchestrator ────────────────────────────────────────────────────────
/**
 * Entry point. Authenticates with GEE, ensures the asset folder exists,
 * then processes all study regions sequentially.
 */
async function main() {
  // ── 1. Authenticate ───────────────────────────────────────────────────────
  try {
    await authenticateGEE();
  } catch (authError) {
    logger.error(`Fatal: GEE authentication failed. ${authError.message}`);
    process.exit(1);
  }

  // ── 2. Ensure the asset destination folder exists ─────────────────────────
  // This is a one-time setup step. On subsequent runs the folder already
  // exists and createAssetFolderIfMissing() resolves immediately.
  try {
    await createAssetFolderIfMissing(ASSET_ROOT);
  } catch (folderError) {
    logger.error(`Fatal: Asset folder setup failed. ${folderError.message}`);
    process.exit(1);
  }

  // ── 3. Resolve date range ─────────────────────────────────────────────────
  const currentYear = new Date().getFullYear();
  const startDate   = process.env.FETCH_START_DATE || `${currentYear - 1}-01-01`;
  const endDate     = process.env.FETCH_END_DATE   || `${currentYear - 1}-12-31`;

  logger.info(`\nDate range  : ${startDate} → ${endDate}`);
  logger.info(`Asset root  : ${ASSET_ROOT}`);
  logger.info(`Export scale: ${EXPORT_SCALE_METRES}m | CRS: ${EXPORT_CRS}`);
  logger.info(`Regions     : ${Object.keys(STUDY_REGIONS).join(", ")}\n`);

  // ── 4. Process each study region ──────────────────────────────────────────
  const allResults = [];
  const allErrors  = [];

  for (const [key, region] of Object.entries(STUDY_REGIONS)) {
    try {
      const result = await fetchSentinel2Composite({
        bbox      : region.bbox,
        startDate,
        endDate,
        regionName: region.name,
      });
      allResults.push({ regionKey: key, ...result });
    } catch (regionError) {
      logger.error(`Fatal error for region "${region.name}": ${regionError.message}`);
      allErrors.push({ regionKey: key, error: regionError.message });
    }
  }

  // ── 5. Final asset manifest ───────────────────────────────────────────────
  const totalTiles    = allResults.reduce((s, r) => s + r.tileCount,    0);
  const totalExported = allResults.reduce((s, r) => s + r.successCount, 0);
  const totalSkipped  = allResults.reduce((s, r) => s + r.skippedCount, 0);
  const totalFailed   = allResults.reduce((s, r) => s + r.failedCount,  0);

  logger.info(`\n${"═".repeat(60)}`);
  logger.info("SENTINEL-2 ASSET EXPORT MANIFEST");
  logger.info(`${"═".repeat(60)}`);
  logger.info(`  Asset root    : ${ASSET_ROOT}`);
  logger.info(`  Date range    : ${startDate} → ${endDate}`);
  logger.info(`  Resolution    : ${EXPORT_SCALE_METRES}m (${EXPORT_CRS})`);
  logger.info(`  Total tiles   : ${totalTiles}`);
  logger.info(`  ✓ Exported    : ${totalExported}`);
  logger.info(`  ⚠ Skipped     : ${totalSkipped}  (already exists or no imagery)`);
  logger.info(`  ✗ Failed      : ${totalFailed}`);
  logger.info(`${"═".repeat(60)}`);
  logger.info(`  Assets written to:`);

  allResults.forEach((r) => {
    logger.info(`\n  Region: ${r.regionName}`);
    r.exportResults
      .filter((t) => t.state === "COMPLETED")
      .forEach((t) => {
        logger.info(`    ✓ ${t.assetId}`);
        logger.info(`      Task ID : ${t.taskId}`);
        logger.info(`      Scenes  : ${t.sceneCount}`);
      });
    r.errors.forEach((e) => {
      logger.warn(`    ✗ ${e.tileId} — ${e.error}`);
    });
  });

  logger.info(
    `\n  Load any asset in GEE via: ` +
    `ee.Image("${ASSET_ROOT}/<tileName>")`
  );
  logger.info(`${"═".repeat(60)}\n`);

  if (allErrors.length === Object.keys(STUDY_REGIONS).length) {
    logger.error("All regions failed. Exiting with error code.");
    process.exit(1);
  }
}

// ─── Exports ──────────────────────────────────────────────────────────────────
export {
  fetchSentinel2Composite,
  submitExportTask,
  pollExportTask,
  createAssetFolderIfMissing,
  assetExists,
  generateTiles,
  validateBoundingBox,
  estimateAreaKm2,
  sanitiseAssetName,
  STUDY_REGIONS,
  TARGET_BANDS,
  S2_COLLECTION_ID,
  MAX_CLOUD_PERCENTAGE,
  MAX_TILE_AREA_KM2,
  ASSET_ROOT,
  EXPORT_SCALE_METRES,
  EXPORT_CRS,
};

// Run main() only when executed directly, not when imported as a module.
const isMain =
  process.argv[1] &&
  fileURLToPath(import.meta.url) === path.resolve(process.argv[1]);

if (isMain) {
  main().catch((err) => {
    logger.error(`Unhandled error in main(): ${err.message}`);
    process.exit(1);
  });
}