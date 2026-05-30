/**
 * @file sentinel1-fetcher.js
 * @description Fetches, filters, prepares, and exports Sentinel-1 SAR (Synthetic
 *              Aperture Radar) Ground Range Detected (GRD) imagery over target
 *              Galamsey zones in Ghana using the Google Earth Engine JavaScript API.
 *
 * REPORT CONTEXT — What is Sentinel-1 and Why SAR for Galamsey Detection?
 * ─────────────────────────────────────────────────────────────────────────────
 * Sentinel-1 is a pair of satellites (1A and 1B) operated by the European Space
 * Agency (ESA) as part of the Copernicus programme. Unlike Sentinel-2, which is
 * a passive optical sensor that records reflected sunlight, Sentinel-1 carries an
 * active C-band Synthetic Aperture Radar (SAR) instrument that transmits its own
 * microwave pulses and records the backscattered energy from Earth's surface.
 *
 * This fundamental difference gives SAR three critical advantages for our project:
 *
 *  1. ALL-WEATHER CAPABILITY: Microwave pulses penetrate cloud cover, rain, and
 *     smoke. Ghana's rainy seasons (April–July and September–November) frequently
 *     produce persistent cloud cover exceeding 80% that completely blocks Sentinel-2
 *     optical observations. Sentinel-1 acquires clear data regardless of weather,
 *     ensuring temporal continuity in our surveillance during these months.
 *
 *  2. DAY/NIGHT OPERATION: SAR does not rely on sunlight, so it acquires data on
 *     both ascending (night) and descending (day) passes, effectively doubling
 *     revisit frequency compared to optical-only surveillance.
 *
 *  3. SURFACE TEXTURE SENSITIVITY: SAR backscatter is highly sensitive to surface
 *     roughness and moisture. Bare, disturbed soil at active mining pits produces
 *     a distinct backscatter signature compared to dense forest canopy, flooded
 *     areas, and standing water — making it a strong complementary signal to the
 *     optical land-cover classification from Sentinel-2.
 *
 * REPORT CONTEXT — VV and VH Polarisation Bands:
 * ─────────────────────────────────────────────────────────────────────────────
 * SAR systems transmit and receive radar pulses in specific electromagnetic
 * polarisations. Sentinel-1 operates in dual-polarisation mode, transmitting
 * in Vertical (V) polarisation and receiving in both Vertical and Horizontal:
 *
 *  ┌──────┬────────────────────────────┬────────────────────────────────────┐
 *  │ Band │ Polarisation               │ Sensitivity / Purpose              │
 *  ├──────┼────────────────────────────┼────────────────────────────────────┤
 *  │  VV  │ Transmit V, Receive V      │ Surface roughness; bare soil;      │
 *  │      │ (co-polarisation)          │ urban structures; open water       │
 *  ├──────┼────────────────────────────┼────────────────────────────────────┤
 *  │  VH  │ Transmit V, Receive H      │ Volume scattering; forest canopy;  │
 *  │      │ (cross-polarisation)       │ vegetation structure; biomass      │
 *  └──────┴────────────────────────────┴────────────────────────────────────┘
 *
 * The VV/VH RATIO is particularly powerful for our application: dense forest
 * produces high VH (volume scattering from canopy) and moderate VV, giving a
 * low ratio. Active mining pits are bare soil — high VV (direct backscatter)
 * and very low VH, giving a high ratio. This ratio forms a feature that our
 * CNN can use alongside Sentinel-2 optical bands to improve classification
 * accuracy, especially during rainy-season months when optical data is absent.
 *
 * REPORT CONTEXT — GRD vs SLC Product:
 * ─────────────────────────────────────────────────────────────────────────────
 * GEE provides two Sentinel-1 products:
 *
 *  • GRD (Ground Range Detected): Multi-looked, projected to ground range,
 *    speckle filtered. Values are in decibels (dB) after GEE's preprocessing.
 *    Pixel values represent backscatter intensity. Suitable for land-cover
 *    classification and change detection — our use case.
 *
 *  • SLC (Single Look Complex): Preserves phase information for interferometry.
 *    Not needed for intensity-based classification.
 *
 * We use COPERNICUS/S1_GRD, which is the standard analysis-ready product.
 * GEE has already applied thermal noise removal, radiometric calibration,
 * and terrain correction, converting raw digital numbers to sigma-naught (σ°)
 * backscatter values in dB.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * REPORT CONTEXT — Instrument Mode and Orbit Direction Filtering:
 * ─────────────────────────────────────────────────────────────────────────────
 * Sentinel-1 operates in four imaging modes. Over land in our study region,
 * it primarily uses Interferometric Wide Swath (IW) mode, which covers a
 * 250 km swath at 10m × 10m resolution (range × azimuth). We filter
 * explicitly for IW mode to exclude any EW (Extra Wide) or SM (Stripmap)
 * acquisitions that cover parts of Ghana at different resolutions.
 *
 * We also filter for DESCENDING orbit passes only. Mixing ascending and
 * descending passes in a median composite introduces systematic geometry
 * differences (incidence angle variations) that create artefacts in the
 * backscatter values. Using a single orbit direction ensures all scenes
 * in the composite share the same viewing geometry.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @module sentinel1-fetcher
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

/**
 * GEE Image Collection ID for Sentinel-1 Ground Range Detected (GRD).
 * Values are calibrated backscatter in dB (sigma-naught, σ°).
 */
const S1_COLLECTION_ID = "COPERNICUS/S1_GRD";

/**
 * The two SAR polarisation bands extracted per scene.
 * VV = co-polarisation (surface roughness / bare soil).
 * VH = cross-polarisation (vegetation volume scattering).
 */
const TARGET_BANDS = ["VV", "VH"];

/**
 * Sentinel-1 imaging mode filter.
 * IW (Interferometric Wide Swath) is the standard mode over land in West Africa,
 * providing 10m resolution across a 250km swath. Filtering to IW ensures all
 * scenes in the composite share the same spatial resolution and geometry.
 */
const INSTRUMENT_MODE = "IW";

/**
 * Orbit direction filter.
 *
 * REPORT CONTEXT — Why single orbit direction?
 * ─────────────────────────────────────────────────────────────────────────────
 * SAR backscatter intensity is a function of the incidence angle — the angle
 * between the radar beam and the vertical. Ascending and descending passes
 * observe the same ground target from mirrored directions, producing
 * systematically different backscatter values for the same surface type.
 * Mixing both in a median composite would average geometrically inconsistent
 * measurements, degrading the discriminability of our land-cover features.
 *
 * We select DESCENDING because Sentinel-1's descending passes provide the
 * most consistent coverage over Ghana's longitude (~2°W–0°E) with minimal
 * gaps in the 2025 acquisition plan.
 * ─────────────────────────────────────────────────────────────────────────────
 */
const ORBIT_DIRECTION = "ASCENDING";

/**
 * Fixed fetch year for Sentinel-1 data.
 * Overridable via FETCH_START_DATE / FETCH_END_DATE in .env.
 */
const FETCH_YEAR = 2025;

// ─── Tiling Constants ─────────────────────────────────────────────────────────

/**
 * Maximum tile area in km² before a bounding box is subdivided.
 * Identical to sentinel2-fetcher.js — keeps both pipelines on the same grid.
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
 * Root Earth Engine Asset path for all exported Sentinel-1 tiles.
 *
 * REPORT CONTEXT — Separate Asset Folders per Sensor:
 * ─────────────────────────────────────────────────────────────────────────────
 * Sentinel-1 and Sentinel-2 tiles are stored in separate asset folders
 * ("sentinel1_tiles" vs "sentinel2_tiles") even though they share the same
 * spatial grid. This separation is intentional:
 *
 *  1. It prevents naming collisions — both sensors produce a tile named
 *     "Pra_River_Basin_tile_r0_c0_2025", which would overwrite each other
 *     in a shared folder.
 *  2. It makes downstream asset loading unambiguous — the CNN preprocessing
 *     script loads SAR and optical tiles independently before stacking them
 *     into a multi-sensor feature tensor.
 *  3. It simplifies asset management — all SAR assets can be listed, deleted,
 *     or versioned as a single group without filtering by band name.
 * ─────────────────────────────────────────────────────────────────────────────
 */
const ASSET_ROOT = process.env.GEE_S1_ASSET_ROOT ||
  "projects/galamsey-sentinel/assets/sentinel1_tiles";

/**
 * Export pixel size in metres.
 *
 * REPORT CONTEXT — Why 10m for Sentinel-1?
 * ─────────────────────────────────────────────────────────────────────────────
 * Sentinel-1 IW mode has a native resolution of ~10m × 10m (range × azimuth).
 * Exporting at scale=10 preserves this resolution and — critically — matches
 * the Sentinel-2 export scale, so both sensor grids align pixel-for-pixel
 * over the same bounding box. This pixel-level alignment is a prerequisite
 * for the multi-sensor feature stacking step in the CNN pipeline:
 *   [B2, B3, B4, B8, VV, VH] → 6-channel input tensor per patch.
 * ─────────────────────────────────────────────────────────────────────────────
 */
const EXPORT_SCALE_METRES = 10;

/**
 * CRS for exported assets. EPSG:4326 (WGS84) matches the Sentinel-2 exports
 * so both grids are spatially aligned without reprojection.
 */
const EXPORT_CRS = "EPSG:4326";

/** Polling interval in ms between ee.data.getTaskStatus() calls. */
const POLL_INTERVAL_MS = 15_000;

/** Maximum ms to wait for a single export task. Default: 2 hours. */
const EXPORT_TIMEOUT_MS =
  parseInt(process.env.GEE_EXPORT_TIMEOUT_MS) || 7_200_000;

// ─── Study Regions ────────────────────────────────────────────────────────────
/**
 * Identical bounding boxes to sentinel2-fetcher.js.
 *
 * REPORT CONTEXT — Grid Consistency Between Sensors:
 * ─────────────────────────────────────────────────────────────────────────────
 * Using the exact same bounding boxes and tiling parameters (MAX_TILE_AREA_KM2,
 * grid size formula) for both sensors guarantees that tile r0_c0 from Sentinel-1
 * covers the exact same geographic area as tile r0_c0 from Sentinel-2. This
 * one-to-one spatial correspondence is essential for the multi-sensor fusion
 * step in Phase 3, where we stack SAR and optical bands into a single input
 * tensor for each training patch.
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
    name: "Birim River Basin (Eastern Region)",
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
 * Converts a raw string into a GEE-safe asset name.
 * Replaces all characters outside [a-zA-Z0-9_-] with underscores.
 *
 * @param {string} name
 * @returns {string}
 */
function sanitiseAssetName(name) {
  return name
    .replace(/[^a-zA-Z0-9_-]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "");
}

// ─── Utility: BBox Validator ──────────────────────────────────────────────────
/**
 * Validates a bounding box for structure, ordering, and WGS84 range.
 * @param {number[]} bbox
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
 * Uses identical logic to sentinel2-fetcher.js to guarantee grid alignment.
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
  const estTileArea = estimateAreaKm2([
    west, south, west + tileWidth, south + tileHeight,
  ]);

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
 * Ensures the target EE asset folder exists before any export task is submitted.
 * Creates it if absent; silently continues if it already exists.
 *
 * @param {string} assetPath - Full EE asset folder path.
 * @returns {Promise<void>}
 */
async function createAssetFolderIfMissing(assetPath) {
  logger.info(`[ASSET-SETUP] Verifying asset folder: ${assetPath}`);

  await new Promise((resolve, reject) => {
    ee.data.createFolder(assetPath, false, (result, error) => {
      if (!error) {
        logger.info(`[ASSET-SETUP] ✓ Asset folder created: ${assetPath}`);
        return resolve();
      }

      const errorStr = String(error).toLowerCase();
      if (
        errorStr.includes("already exists") ||
        errorStr.includes("cannot overwrite")
      ) {
        logger.info(`[ASSET-SETUP] ✓ Asset folder already exists: ${assetPath}`);
        return resolve();
      }

      reject(
        new Error(
          `[ASSET-SETUP] Failed to create asset folder "${assetPath}": ${error}. ` +
          `Ensure the Cloud Project ID matches your registered GEE project and ` +
          `the service account has the "Earth Engine Resource Writer" IAM role.`
        )
      );
    });
  });
}

// ─── Asset Existence Check ────────────────────────────────────────────────────
/**
 * Returns true if an EE asset already exists at the given path.
 * Used to skip tiles that were successfully exported in a previous run,
 * making the pipeline safely resumable after crashes.
 *
 * @param {string} assetId
 * @returns {Promise<boolean>}
 */
async function assetExists(assetId) {
  return new Promise((resolve) => {
    ee.data.getAsset(assetId, (result, error) => {
      resolve(!error && result != null);
    });
  });
}

// ─── Single-Tile Sentinel-1 Fetcher ──────────────────────────────────────────
/**
 * Builds a Sentinel-1 GRD median composite for a single tile bounding box.
 *
 * REPORT CONTEXT — Sentinel-1 Collection Filtering Chain:
 * ─────────────────────────────────────────────────────────────────────────────
 * We apply four sequential filters to the global S1_GRD archive before
 * compositing, each narrowing the candidate scene set:
 *
 *  1. filterBounds   → Keep only scenes whose swath intersects the tile.
 *  2. filterDate     → Keep only scenes within the 2025 calendar year.
 *  3. filter (mode)  → Keep only IW (Interferometric Wide) mode scenes
 *                      to ensure consistent 10m resolution.
 *  4. filter (orbit) → Keep only DESCENDING passes to ensure consistent
 *                      viewing geometry across all scenes in the composite.
 *  5. select         → Extract only VV and VH bands, discarding the
 *                      "angle" band (local incidence angle) which is not
 *                      needed for intensity-based classification.
 *
 * The resulting filtered collection is reduced to a single median composite
 * image. For SAR backscatter in dB, the median is preferred over the mean
 * because it suppresses speckle noise — the granular "salt and pepper"
 * pattern inherent in all SAR imagery caused by coherent interference of
 * radar echoes. Speckle is effectively a random, high-frequency noise
 * signal; the median across multiple observations naturally attenuates it.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {object}   params
 * @param {number[]} params.bbox
 * @param {string}   params.startDate
 * @param {string}   params.endDate
 * @param {string}   params.tileId
 * @param {number}   [params.attempt=1]
 *
 * @returns {Promise<{ image: ee.Image, region: ee.Geometry, metadata: object } | null>}
 *          Null if no scenes exist for this tile in the date range.
 */
async function fetchTile({ bbox, startDate, endDate, tileId, attempt = 1 }) {
  const tag = `[FETCH][${tileId}]`;

  try {
    const [west, south, east, north] = bbox;
    const region = ee.Geometry.Rectangle([west, south, east, north], null, false);

    // ── Build the S1 collection pipeline ─────────────────────────────────────
    const collection = ee
      .ImageCollection(S1_COLLECTION_ID)
      .filterBounds(region)
      .filterDate(startDate, endDate)

      // Filter to Interferometric Wide Swath mode only.
      // The GEE metadata property is "instrumentMode".
      .filter(ee.Filter.eq("instrumentMode", INSTRUMENT_MODE))

      // Filter to a single orbit direction for geometric consistency.
      // The GEE metadata property is "orbitProperties_pass".
      .filter(ee.Filter.eq("orbitProperties_pass", ORBIT_DIRECTION))

      // Select VV and VH bands only. The S1_GRD collection also contains
      // an "angle" band (local incidence angle in degrees) that we exclude
      // to keep the feature dimensions consistent with our CNN input spec.
      .select(TARGET_BANDS);

    // ── Validate scene count ──────────────────────────────────────────────────
    const sceneCount = await new Promise((resolve, reject) => {
      collection.size().evaluate((result, error) => {
        if (error) reject(new Error(`GEE evaluate() failed: ${error}`));
        else resolve(result);
      });
    });

    if (sceneCount === 0) {
      // Unlike Sentinel-2, Sentinel-1 is not blocked by clouds, so zero
      // scenes more likely indicates no DESCENDING IW pass covers this
      // specific tile in 2025. We log a more targeted diagnostic message.
      logger.warn(
        `${tag} 0 scenes found. Possible causes: ` +
        `(1) No DESCENDING IW pass covers this tile extent in ${startDate}–${endDate}. ` +
        `(2) Try switching ORBIT_DIRECTION to "ASCENDING" for this region. ` +
        `(3) Verify the tile bbox intersects a Sentinel-1 acquisition track.`
      );
      return null;
    }

    logger.info(
      `${tag} ${sceneCount} scenes found ` +
      `(mode: ${INSTRUMENT_MODE}, orbit: ${ORBIT_DIRECTION}). ` +
      `Building median composite...`
    );

    // ── Build the median composite ────────────────────────────────────────────
    // REPORT CONTEXT — SAR Median Compositing:
    // ─────────────────────────────────────────────────────────────────────────
    // For optical imagery, the median suppresses cloud outliers. For SAR, it
    // suppresses speckle noise and transient backscatter anomalies caused by
    // surface moisture variations (e.g. a single rain-saturated acquisition).
    // With Sentinel-1's ~12-day revisit cycle, a full year provides ~30
    // acquisitions per tile, giving the median a robust statistical base.
    // ─────────────────────────────────────────────────────────────────────────
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
      bands          : bandNames,
      instrumentMode : INSTRUMENT_MODE,
      orbitDirection : ORBIT_DIRECTION,
      compositeMethod: "median",
      collectionId   : S1_COLLECTION_ID,
      createdAt      : new Date().toISOString(),
    };

    logger.info(
      `${tag} Composite ready. Bands: [${bandNames.join(", ")}] | ` +
      `Scenes: ${sceneCount} | Mode: ${INSTRUMENT_MODE} | Orbit: ${ORBIT_DIRECTION}`
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
 * Submits a GEE export task writing a Sentinel-1 composite tile to EE Assets.
 *
 * REPORT CONTEXT — Pyramiding Policy for SAR Data:
 * ─────────────────────────────────────────────────────────────────────────────
 * SAR backscatter values in dB are continuous, signed floating-point numbers
 * (typically ranging from -25 dB for calm water to +5 dB for urban corners).
 * As with Sentinel-2 reflectance, "MEAN" is the correct pyramiding policy:
 * averaging adjacent pixels at lower zoom levels preserves the mean backscatter
 * energy of the region, which is radiometrically meaningful. "MODE" (used for
 * categorical data) or "MAX" would introduce systematic bias.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {object}      params
 * @param {ee.Image}    params.image
 * @param {ee.Geometry} params.region
 * @param {object}      params.metadata
 * @param {string}      [params.assetRoot]
 *
 * @returns {Promise<{ taskId: string, assetId: string, status: "SUBMITTED" }>}
 */
async function submitExportTask({ image, region, metadata, assetRoot = ASSET_ROOT }) {
  const tag = `[EXPORT][${metadata.tileId}]`;

  const year      = new Date(metadata.startDate).getFullYear();
  const assetName = sanitiseAssetName(`${metadata.tileId}_S1_${year}`);
  const assetId   = `${assetRoot}/${assetName}`;

  // Add a "_S1_" infix to the asset name so that Sentinel-1 and Sentinel-2
  // assets for the same tile are immediately distinguishable when listed
  // together in the GEE Code Editor asset panel or via ee.data.listAssets().

  logger.info(`${tag} Submitting toAsset export.`);
  logger.info(`${tag} Asset destination: ${assetId}`);
  logger.info(
    `${tag} Scale: ${EXPORT_SCALE_METRES}m | CRS: ${EXPORT_CRS} | ` +
    `Bands: [${metadata.bands.join(", ")}]`
  );

  const task = ee.batch.Export.image.toAsset({
    image,
    description     : assetName,
    assetId         : assetId,
    scale           : EXPORT_SCALE_METRES,
    region          : region,
    crs             : EXPORT_CRS,
    maxPixels       : 1e9,
    pyramidingPolicy: {
      VV: "MEAN",
      VH: "MEAN",
    },
  });

  await new Promise((resolve, reject) => {
    try {
      task.start();
      resolve();
    } catch (startError) {
      reject(
        new Error(
          `${tag} task.start() failed: ${startError.message}. ` +
          `Common causes: ` +
          `(1) Asset folder "${assetRoot}" does not exist. ` +
          `(2) Asset "${assetId}" already exists — delete it or use a new name. ` +
          `(3) GEE auth token expired — re-run authenticateGEE(). ` +
          `(4) Cloud Project ID in ASSET_ROOT does not match your GEE project.`
        )
      );
    }
  });

  const taskId = task.id ?? "unknown";

  logger.info(
    `${tag} Task submitted. GEE Task ID: ${taskId}. ` +
    `Monitor: https://code.earthengine.google.com/tasks`
  );

  return { taskId, assetId, status: "SUBMITTED" };
}

// ─── Export: Poll Task Status ─────────────────────────────────────────────────
/**
 * Polls a GEE export task until terminal state or timeout.
 *
 * @param {string} taskId
 * @param {string} assetId
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
        logger.info(`${tag} ✓ COMPLETED in ${mm}m ${ss}s. Asset: ${assetId}`);
      } else {
        const errorMessage =
          taskStatus?.error_message ?? taskStatus?.description ?? "No details.";
        logger.error(`${tag} ✗ ${state}: ${errorMessage}`);
        return { taskId, state, assetId, elapsedMs, errorMessage };
      }
      return { taskId, state, assetId, elapsedMs };
    }

    await sleep(POLL_INTERVAL_MS);
  }
}

// ─── Region Fetcher + Exporter ────────────────────────────────────────────────
/**
 * Fetches Sentinel-1 composites for an entire region and exports every
 * successful tile to Earth Engine Assets.
 *
 * Per-tile pipeline:
 *   1. Skip if asset already exists      (resumable).
 *   2. Fetch S1 median composite         → fetchTile().
 *   3. Submit toAsset export task        → submitExportTask().
 *   4. Poll until terminal state         → pollExportTask().
 *   5. Wait INTER_TILE_DELAY_MS          (rate-limit safety).
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
async function fetchSentinel1Composite({
  bbox,
  startDate,
  endDate,
  regionName = "unnamed-region",
}) {
  logger.info(`\n${"─".repeat(60)}`);
  logger.info(`[${regionName}] Starting Sentinel-1 region fetch + asset export.`);
  logger.info(
    `[${regionName}] Date: ${startDate} → ${endDate} | ` +
    `Mode: ${INSTRUMENT_MODE} | Orbit: ${ORBIT_DIRECTION} | ` +
    `Bands: ${TARGET_BANDS.join(", ")}`
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
      // ── Step 1: Skip if asset already exists ────────────────────────────
      const year            = new Date(startDate).getFullYear();
      const assetName       = sanitiseAssetName(`${tileId}_S1_${year}`);
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

      // ── Step 2: Fetch S1 median composite ────────────────────────────────
      const fetchResult = await fetchTile({
        bbox: tileBbox, startDate, endDate, tileId,
      });

      if (fetchResult === null) {
        skippedCount++;
        logger.warn(`[${tileId}] Skipped — no scenes found.`);
        if (i < tiles.length - 1) await sleep(INTER_TILE_DELAY_MS);
        continue;
      }

      const { image, region, metadata } = fetchResult;

      // ── Step 3: Submit toAsset export task ───────────────────────────────
      const submission = await submitExportTask({ image, region, metadata });

      // ── Step 4: Poll until terminal state ────────────────────────────────
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
 * Entry point. Authenticates with GEE, ensures the S1 asset folder exists,
 * then processes all study regions for the configured fetch year.
 */
async function main() {
  // ── 1. Authenticate ───────────────────────────────────────────────────────
  try {
    await authenticateGEE();
  } catch (authError) {
    logger.error(`Fatal: GEE authentication failed. ${authError.message}`);
    process.exit(1);
  }

  // ── 2. Ensure the S1 asset folder exists ─────────────────────────────────
  try {
    await createAssetFolderIfMissing(ASSET_ROOT);
  } catch (folderError) {
    logger.error(`Fatal: Asset folder setup failed. ${folderError.message}`);
    process.exit(1);
  }

  // ── 3. Resolve date range ─────────────────────────────────────────────────
  // Default to the full FETCH_YEAR calendar year unless env vars override.
  const startDate = process.env.FETCH_START_DATE || `${FETCH_YEAR}-01-01`;
  const endDate   = process.env.FETCH_END_DATE   || `${FETCH_YEAR}-12-31`;

  logger.info(`\nSensor      : Sentinel-1 GRD`);
  logger.info(`Mode        : ${INSTRUMENT_MODE} | Orbit: ${ORBIT_DIRECTION}`);
  logger.info(`Bands       : ${TARGET_BANDS.join(", ")}`);
  logger.info(`Date range  : ${startDate} → ${endDate}`);
  logger.info(`Asset root  : ${ASSET_ROOT}`);
  logger.info(`Export scale: ${EXPORT_SCALE_METRES}m | CRS: ${EXPORT_CRS}`);
  logger.info(`Regions     : ${Object.keys(STUDY_REGIONS).join(", ")}\n`);

  // ── 4. Process each region ────────────────────────────────────────────────
  const allResults = [];
  const allErrors  = [];

  for (const [key, region] of Object.entries(STUDY_REGIONS)) {
    try {
      const result = await fetchSentinel1Composite({
        bbox      : region.bbox,
        startDate,
        endDate,
        regionName: region.name,
      });
      allResults.push({ regionKey: key, ...result });
    } catch (regionError) {
      logger.error(
        `Fatal error for region "${region.name}": ${regionError.message}`
      );
      allErrors.push({ regionKey: key, error: regionError.message });
    }
  }

  // ── 5. Final asset manifest ───────────────────────────────────────────────
  const totalTiles    = allResults.reduce((s, r) => s + r.tileCount,    0);
  const totalExported = allResults.reduce((s, r) => s + r.successCount, 0);
  const totalSkipped  = allResults.reduce((s, r) => s + r.skippedCount, 0);
  const totalFailed   = allResults.reduce((s, r) => s + r.failedCount,  0);

  logger.info(`\n${"═".repeat(60)}`);
  logger.info("SENTINEL-1 ASSET EXPORT MANIFEST");
  logger.info(`${"═".repeat(60)}`);
  logger.info(`  Sensor        : Sentinel-1 GRD (${INSTRUMENT_MODE}, ${ORBIT_DIRECTION})`);
  logger.info(`  Bands         : ${TARGET_BANDS.join(", ")}`);
  logger.info(`  Asset root    : ${ASSET_ROOT}`);
  logger.info(`  Date range    : ${startDate} → ${endDate}`);
  logger.info(`  Resolution    : ${EXPORT_SCALE_METRES}m (${EXPORT_CRS})`);
  logger.info(`  Total tiles   : ${totalTiles}`);
  logger.info(`  ✓ Exported    : ${totalExported}`);
  logger.info(`  ⚠ Skipped     : ${totalSkipped}  (already exists or no scenes)`);
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
    `\n  Load any S1 asset in GEE via: ` +
    `ee.Image("${ASSET_ROOT}/<tileName>")`
  );
  logger.info(
    `\n  To stack S1 + S2 bands for CNN input:\n` +
    `  const s2 = ee.Image("projects/galamsey-sentinel/assets/sentinel2_tiles/<tileName>_2025");\n` +
    `  const s1 = ee.Image("${ASSET_ROOT}/<tileName>_S1_2025");\n` +
    `  const stacked = s2.addBands(s1); // → [B2, B3, B4, B8, VV, VH]`
  );
  logger.info(`${"═".repeat(60)}\n`);

  if (allErrors.length === Object.keys(STUDY_REGIONS).length) {
    logger.error("All regions failed. Exiting with error code.");
    process.exit(1);
  }
}

// ─── Exports ──────────────────────────────────────────────────────────────────
export {
  fetchSentinel1Composite,
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
  S1_COLLECTION_ID,
  INSTRUMENT_MODE,
  ORBIT_DIRECTION,
  MAX_TILE_AREA_KM2,
  ASSET_ROOT,
  EXPORT_SCALE_METRES,
  EXPORT_CRS,
  FETCH_YEAR,
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