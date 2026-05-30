/**
 * @file patch-extractor.js
 * @description Phase 3 — Patch Extraction and Ground Truth Labeling.
 *              Loads 8-band processed feature composites from Earth Engine Assets,
 *              extracts a 1280m × 1280m GeoTIFF patch around each labelled
 *              ground-truth point, and saves them into a class-partitioned
 *              folder structure compatible with PyTorch ImageFolder and
 *              TensorFlow ImageDataGenerator.
 *
 * OUTPUT STRUCTURE:
 *   pipeline/extraction/patches/
 *     class_0_forest/
 *       forest_001.tif
 *       forest_002.tif
 *     class_1_galamsey/
 *       galamsey_001.tif
 *       galamsey_002.tif
 *     class_2_water/
 *       water_001.tif
 *
 * REPORT CONTEXT — Why GeoTIFF + ImageFolder Structure?
 * ─────────────────────────────────────────────────────────────────────────────
 * After testing revealed that GEE's FeatureCollection download API does not
 * support TFRecord format (it only supports CSV/GeoJSON/KML/SHP), we pivot
 * to the standard computer vision dataset convention:
 *
 *   One file per sample, organised into per-class subdirectories.
 *
 * This format is natively supported by:
 *   • PyTorch  : torchvision.datasets.ImageFolder("patches/")
 *   • TensorFlow/Keras : tf.keras.utils.image_dataset_from_directory("patches/")
 *   • TF.js    : Can be loaded via a custom GeoTIFF → tensor loader using geotiff.js
 *
 * GeoTIFF retains full geospatial metadata (CRS, pixel size, bounding box)
 * alongside the pixel values, which means patches can be reprojected,
 * inspected in QGIS, and used for spatial cross-validation without any
 * loss of positional context.
 *
 * REPORT CONTEXT — Patch Dimensions:
 * ─────────────────────────────────────────────────────────────────────────────
 * We buffer the ground-truth point by 640m and take the bounding box of the
 * resulting circle, producing a roughly 1280m × 1280m square region.
 * At Sentinel-2's 10m native resolution this yields approximately:
 *   1280m / 10m = 128 pixels per side → 128 × 128 × 8 bands
 *
 * "Approximately" because:
 *   • ee.Geometry.Point.buffer(640) creates a geodesic circle, not a square.
 *   • .bounds() returns the minimum bounding rectangle of that circle.
 *   • At Ghana's latitude (~5–6°N) the longitude/latitude degree lengths
 *     differ slightly, so the bounding box is very close to but not exactly
 *     1280m × 1280m. The actual pixel count is typically 127–129 px per side.
 *   • GEE resamples to the nearest pixel boundary at export time.
 *
 * This is acceptable for CNN training — standard augmentation pipelines
 * include random crops and resizes that handle ±1–2 pixel variation.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * REPORT CONTEXT — ee.Image.getDownloadURL() vs Export.image.toDrive():
 * ─────────────────────────────────────────────────────────────────────────────
 * getDownloadURL() is a synchronous, size-bounded download path:
 *   • GEE computes the image on-demand and returns a signed HTTPS URL.
 *   • Our process downloads it directly — no Drive, no Cloud Storage needed.
 *   • Practical size limit: ~32 MB per call (enforced by GEE server).
 *   • At 128×128 × 8 bands × float32 (4 bytes) ≈ 524 KB per patch — well
 *     within the limit. Even at 50 points the total is ~26 MB.
 *   • The URL is valid for ~2 hours after generation.
 *
 * @module patch-extractor
 */

import ee                            from "@google/earthengine";
import path                          from "path";
import fs                            from "fs/promises";
import { createWriteStream }         from "fs";
import { fileURLToPath }             from "url";
import { pipeline as streamPipeline } from "stream/promises";
import "dotenv/config";
import { authenticateGEE, logger }   from "../ingestion/gee-auth.js";

// ─── ES Module __dirname Shim ─────────────────────────────────────────────────
const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 1 — GROUND TRUTH DICTIONARY
//
// HOW TO ADD MORE POINTS:
// ─────────────────────────────────────────────────────────────────────────────
// Each entry in a class array is an object with three fields:
//
//   coords : [longitude, latitude]  ← WGS84 decimal degrees. Required.
//   id     : "unique_string"        ← Becomes the output filename. Required.
//                                     Use only letters, numbers, underscores.
//   notes  : "free text"            ← Field observation notes. Optional.
//
// To add 50 more Galamsey sites, paste them into the `galamsey` array.
// To add a new class entirely (e.g. degraded_forest):
//   1. Add a new key + array here.
//   2. Add the integer label to CLASS_LABELS below.
//   No other code changes are required.
//
// Class definitions:
//   Class 0 → forest   : intact canopy (negative / non-mining examples)
//   Class 1 → galamsey : active mining pit (positive examples)
//   Class 2 → water    : river / water body surface
// ═══════════════════════════════════════════════════════════════════════════════

const GROUND_TRUTH_POINTS = {

  // ── Class 1: Active Galamsey / Mining Pits ──────────────────────────────
  galamsey: [
    {
      coords : [-1.553105, 5.598837],
      id     : "galamsey_001",
      notes  : "Active pit, Pra River Basin — confirmed from visual inspection",
    },
    // ── Paste additional Class 1 points below this line ───────────────────
    // { coords: [-1.XXXXXX, 5.XXXXXX], id: "galamsey_002", notes: "" },
    // { coords: [-1.XXXXXX, 5.XXXXXX], id: "galamsey_003", notes: "" },
  ],

  // ── Class 0: Intact Forest / Non-Mining Vegetation ───────────────────────
  forest: [
    {
      coords : [-1.510876, 5.615280],
      id     : "forest_001",
      notes  : "Dense canopy, Pra River Basin — confirmed non-mining area",
    },
    // ── Paste additional Class 0 points below this line ───────────────────
    // { coords: [-1.XXXXXX, 5.XXXXXX], id: "forest_002", notes: "" },
    // { coords: [-1.XXXXXX, 5.XXXXXX], id: "forest_003", notes: "" },
  ],

  // ── Class 2: Water Bodies / River Surfaces ────────────────────────────────
  water: [
    // ── Paste Class 2 points below this line ──────────────────────────────
    // { coords: [-1.XXXXXX, 5.XXXXXX], id: "water_001", notes: "Pra River upstream of Tarkwa" },
    // { coords: [-1.XXXXXX, 5.XXXXXX], id: "water_002", notes: "Birim River near Oda" },
  ],

};

/**
 * Maps class category names to integer labels.
 * The CNN's final softmax layer must have exactly this many output neurons.
 */
const CLASS_LABELS = {
  forest   : 0,
  galamsey : 1,
  water    : 2,
};

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 2 — PATCH & PATH CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Buffer radius in metres applied to each ground-truth point.
 * buffer(640) → bounding box ≈ 1280m × 1280m → ~128×128 pixels at 10m.
 *
 * REPORT CONTEXT — Why buffer + bounds instead of a fixed pixel grid?
 * ─────────────────────────────────────────────────────────────────────────────
 * ee.Geometry.Point.buffer(r).bounds() is the idiomatic GEE pattern for
 * extracting a square-ish region around a point in metric units. It is
 * preferable to computing degree offsets manually because:
 *
 *  1. The buffer is defined in metres, not degrees. At Ghana's latitude
 *     (~5°N) one degree of longitude ≈ 110.6 km and one degree of latitude
 *     ≈ 110.9 km — close but not equal. A 640m degree-based offset would
 *     produce a very slightly non-square patch. buffer() is exact.
 *
 *  2. .bounds() returns the axis-aligned minimum bounding rectangle of the
 *     circular buffer, giving a consistent rectangular export region.
 * ─────────────────────────────────────────────────────────────────────────────
 */
const BUFFER_RADIUS_M = 640;

/** Export pixel size in metres — must match Phase 2.5 EXPORT_SCALE_METRES. */
const EXPORT_SCALE_METRES = 10;

/** CRS — must match Phase 2.5 EXPORT_CRS. */
const EXPORT_CRS = "EPSG:4326";

/** Band names — must match Phase 2.5 OUTPUT_BAND_ORDER exactly. */
const BAND_NAMES = ["B2", "B3", "B4", "B8", "VV", "VH", "NDVI", "NDWI"];

/** Source asset root for 8-band feature composites (Phase 2.5 output). */
const FEATURES_ASSET_ROOT = process.env.GEE_FEATURES_ASSET_ROOT ||
  "projects/galamsey-sentinel/assets/processed_features";

/** Data year suffix — must match DATA_YEAR used in feature-engineer.js. */
const DATA_YEAR = parseInt(process.env.DATA_YEAR) || 2025;

/**
 * Root output directory for all downloaded patches.
 * Resolved relative to this script's location:
 *   pipeline/extraction/patches/
 */
const PATCHES_ROOT = path.resolve(__dirname, "patches");

/** Milliseconds to wait between sequential point processing runs. */
const INTER_POINT_DELAY_MS = 2000;

/** Maximum retry attempts per point for transient errors. */
const MAX_RETRIES = 3;

/** Base delay in ms for exponential backoff. Attempt N waits N × base. */
const RETRY_BASE_DELAY_MS = 10_000;

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 3 — STUDY REGIONS & TILING (reproduced for tile-point lookup)
// ═══════════════════════════════════════════════════════════════════════════════

/** Identical to all upstream pipeline scripts — required for tile resolution. */
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

const MAX_TILE_AREA_KM2 = 2000;

function estimateAreaKm2(bbox) {
  const [west, south, east, north] = bbox;
  const midLat = (south + north) / 2;
  const latKm  = (north - south) * 111;
  const lonKm  = (east  - west)  * 111 * Math.cos((midLat * Math.PI) / 180);
  return Math.round(latKm * lonKm);
}

function generateTiles(bbox, regionName) {
  const [west, south, east, north] = bbox;
  const totalArea = estimateAreaKm2(bbox);

  if (totalArea <= MAX_TILE_AREA_KM2) {
    return [{ tileId: `${regionName}_tile_r0_c0`, bbox, row: 0, col: 0 }];
  }

  const tilesNeeded = Math.ceil(totalArea / MAX_TILE_AREA_KM2);
  const gridSize    = Math.ceil(Math.sqrt(tilesNeeded));
  const tileWidth   = (east  - west)  / gridSize;
  const tileHeight  = (north - south) / gridSize;

  const tiles = [];
  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
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

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 4 — UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

/** @param {number} ms @returns {Promise<void>} */
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** @param {string} name @returns {string} filesystem-safe name */
function sanitiseName(name) {
  return name
    .replace(/[^a-zA-Z0-9_-]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "");
}

/**
 * Returns the class-specific subdirectory name for a given class.
 * Format: "class_<label>_<name>"  e.g. "class_1_galamsey"
 *
 * Using both the integer label AND the name in the folder name means:
 *  • PyTorch's ImageFolder can infer the integer label from the folder name
 *    via a simple sort (class_0 < class_1 < class_2).
 *  • The name makes the folder human-readable without needing a lookup table.
 *
 * @param {string} className
 * @returns {string}
 */
function classDirName(className) {
  return `class_${CLASS_LABELS[className]}_${className}`;
}

/**
 * Returns the absolute path to the class subdirectory.
 * @param {string} className
 * @returns {string}
 */
function classDirPath(className) {
  return path.join(PATCHES_ROOT, classDirName(className));
}

/**
 * Returns the absolute path for a single patch GeoTIFF.
 * e.g. pipeline/extraction/patches/class_1_galamsey/galamsey_001.tif
 *
 * @param {string} className
 * @param {string} pointId
 * @returns {string}
 */
function patchFilePath(className, pointId) {
  return path.join(classDirPath(className), `${sanitiseName(pointId)}.tif`);
}

/**
 * Flattens the GROUND_TRUTH_POINTS dictionary into a single ordered array.
 * This is the only function that needs to run when you add more coordinates.
 *
 * @returns {Array<{
 *   coords    : [number, number],
 *   id        : string,
 *   notes     : string,
 *   className : string,
 *   classLabel: number
 * }>}
 */
function flattenGroundTruthPoints() {
  const allPoints = [];

  for (const [className, points] of Object.entries(GROUND_TRUTH_POINTS)) {
    const classLabel = CLASS_LABELS[className];

    if (classLabel === undefined) {
      logger.warn(
        `[SETUP] Class "${className}" not found in CLASS_LABELS. ` +
        `Add it to CLASS_LABELS before adding points of this type.`
      );
      continue;
    }

    for (const point of points) {
      if (!Array.isArray(point.coords) || point.coords.length !== 2) {
        logger.warn(`[SETUP] Point "${point.id}" has invalid coords. Skipping.`);
        continue;
      }
      allPoints.push({
        coords    : point.coords,
        id        : point.id,
        notes     : point.notes ?? "",
        className,
        classLabel,
      });
    }
  }

  return allPoints;
}

/**
 * Creates the root patches directory and one subdirectory per class.
 * Uses recursive: true so the call is idempotent — safe to run on every
 * pipeline start without checking whether directories already exist.
 *
 * @returns {Promise<void>}
 */
async function ensureOutputDirectories() {
  // Create the root patches directory.
  await fs.mkdir(PATCHES_ROOT, { recursive: true });
  logger.info(`[SETUP] Patches root: ${PATCHES_ROOT}`);

  // Create one subdirectory per class defined in CLASS_LABELS.
  for (const className of Object.keys(CLASS_LABELS)) {
    const dirPath = classDirPath(className);
    await fs.mkdir(dirPath, { recursive: true });
    logger.info(`[SETUP]   ${classDirName(className)}/`);
  }
}

/**
 * Returns true if the patch GeoTIFF already exists on disk.
 * Used to make the pipeline resumable — skip files already downloaded.
 *
 * @param {string} filepath - Absolute path to check.
 * @returns {Promise<boolean>}
 */
async function patchFileExists(filepath) {
  try {
    await fs.access(filepath);
    return true;
  } catch {
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 5 — FEATURE COMPOSITE LOADER
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Resolves which processed feature asset covers a given [lon, lat] coordinate
 * by reconstructing the deterministic tile grid from the upstream pipeline
 * scripts, then loads and validates the matching asset.
 *
 * REPORT CONTEXT — Tile resolution strategy:
 * ─────────────────────────────────────────────────────────────────────────────
 * Our feature composites are stored as individual tile assets, not as a
 * single mosaicked image. We must find which tile contains the coordinate
 * before loading it. We reconstruct the same grid used in Phase 1–2.5 by
 * running the identical generateTiles() function with the same parameters —
 * the result is deterministic, so tile IDs and bounding boxes are guaranteed
 * to match the exported assets.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {number} lon     - Longitude (decimal degrees).
 * @param {number} lat     - Latitude (decimal degrees).
 * @param {string} pointId - For log messages.
 * @returns {Promise<{ image: ee.Image, tileId: string, assetId: string }>}
 * @throws {Error} If no tile contains the point or the asset is missing/invalid.
 */
async function loadFeatureCompositeForPoint(lon, lat, pointId) {
  const tag = `[LOAD][${pointId}]`;

  // Reconstruct the full tile inventory across all study regions.
  const allTiles = [];
  for (const [, region] of Object.entries(STUDY_REGIONS)) {
    generateTiles(region.bbox, region.name).forEach((t) => allTiles.push(t));
  }

  // Find the tile whose bounding box contains this coordinate.
  // A small epsilon handles points on tile boundaries.
  const EPSILON      = 0.0001;
  const matchingTile = allTiles.find(({ bbox }) => {
    const [west, south, east, north] = bbox;
    return (
      lon >= west  - EPSILON && lon <= east  + EPSILON &&
      lat >= south - EPSILON && lat <= north + EPSILON
    );
  });

  if (!matchingTile) {
    throw new Error(
      `${tag} Point [${lon}, ${lat}] is outside all study region tiles. ` +
      `Verify the coordinate falls within the Pra, Ankobra, or Birim basin bboxes.`
    );
  }

  logger.info(`${tag} Point [${lon}, ${lat}] → tile: ${matchingTile.tileId}`);

  // Reconstruct the asset ID using the same convention as feature-engineer.js.
  const assetName = sanitiseName(`${matchingTile.tileId}_features_${DATA_YEAR}`);
  const assetId   = `${FEATURES_ASSET_ROOT}/${assetName}`;

  // Confirm the asset exists before constructing the ee.Image.
  const exists = await new Promise((resolve) => {
    ee.data.getAsset(assetId, (result, error) => resolve(!error && result != null));
  });

  if (!exists) {
    throw new Error(
      `${tag} Feature asset not found: "${assetId}". ` +
      `Ensure feature-engineer.js completed successfully for tile "${matchingTile.tileId}".`
    );
  }

  const image = ee.Image(assetId);

  // Validate band availability before building the download URL.
  // This is a cheap server-side check (~1s) that surfaces missing-band
  // errors immediately rather than minutes later during the download.
  const bandNames = await new Promise((resolve, reject) => {
    image.bandNames().evaluate((result, error) => {
      if (error) reject(new Error(`${tag} Band validation failed: ${error}`));
      else resolve(result);
    });
  });

  const missing = BAND_NAMES.filter((b) => !bandNames.includes(b));
  if (missing.length > 0) {
    throw new Error(
      `${tag} Asset "${assetId}" is missing bands: [${missing.join(", ")}]. ` +
      `Re-run feature-engineer.js for tile "${matchingTile.tileId}".`
    );
  }

  logger.info(`${tag} Asset validated: [${bandNames.join(", ")}]`);

  return { image, tileId: matchingTile.tileId, assetId };
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 6 — PATCH BOUNDING BOX
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Builds the 1280m × 1280m GEE geometry bounding box for a ground-truth point.
 *
 * REPORT CONTEXT — Buffer + Bounds Strategy:
 * ─────────────────────────────────────────────────────────────────────────────
 * ee.Geometry.Point([lon, lat])
 *   .buffer(640)    → Creates a geodesic circle of radius 640m around the point.
 *                     Computed on GEE's servers in the geodesic CRS (WGS84).
 *   .bounds()       → Returns the minimum axis-aligned bounding rectangle of
 *                     that circle. This is the geometry we pass to getDownloadURL.
 *
 * The result is a rectangle of approximately 1280m × 1280m (≈128×128px at 10m)
 * centred on the ground-truth point, with sides parallel to the lat/lon axes.
 *
 * We call .getInfo() here to materialise the bounding box coordinates as a
 * plain JavaScript GeoJSON object. This is necessary because getDownloadURL
 * accepts either an ee.Geometry (server-side) or a GeoJSON object (client-side).
 * Using the client-side GeoJSON form avoids an extra server round-trip when
 * the download URL is generated.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {number} lon     - Longitude (decimal degrees).
 * @param {number} lat     - Latitude (decimal degrees).
 * @param {string} pointId - For log messages.
 * @returns {Promise<{ eeGeometry: ee.Geometry, geojson: object, bboxDeg: number[] }>}
 */
async function buildPatchBBox(lon, lat, pointId) {
  const tag = `[BBOX][${pointId}]`;

  // Build the server-side geometry: circle buffer → bounding rectangle.
  const eeGeometry = ee.Geometry.Point([lon, lat])
    .buffer(BUFFER_RADIUS_M)
    .bounds();

  // Materialise the bounding box as a GeoJSON object so we can log it
  // and pass it to getDownloadURL as a plain JS object.
  const geojson = await new Promise((resolve, reject) => {
    eeGeometry.evaluate((result, error) => {
      if (error) {
        reject(new Error(`${tag} Failed to evaluate bounding box geometry: ${error}`));
      } else {
        resolve(result);
      }
    });
  });

  // Extract the [west, south, east, north] coordinates from the GeoJSON bbox.
  // GeoJSON Polygon coordinates are: [[W,S],[E,S],[E,N],[W,N],[W,S]] (ring).
  const coords  = geojson.coordinates[0];
  const west    = coords[0][0];
  const south   = coords[0][1];
  const east    = coords[2][0];
  const north   = coords[2][1];
  const bboxDeg = [west, south, east, north];

  // Estimate the actual pixel dimensions at the export scale.
  // This is informational — GEE determines the true pixel count at export time.
  const widthM  = (east  - west)  * 111_000 * Math.cos((lat * Math.PI) / 180);
  const heightM = (north - south) * 111_000;
  const widthPx = Math.round(widthM  / EXPORT_SCALE_METRES);
  const heightPx = Math.round(heightM / EXPORT_SCALE_METRES);

  logger.info(
    `${tag} BBox: [${bboxDeg.map((v) => v.toFixed(5)).join(", ")}] | ` +
    `Estimated patch: ~${widthPx}×${heightPx}px at ${EXPORT_SCALE_METRES}m`
  );

  return { eeGeometry, geojson, bboxDeg };
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 7 — GETDOWNLOADURL + LOCAL DOWNLOAD
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Requests a signed GeoTIFF download URL from GEE for a specific image patch,
 * then streams the file directly to the correct class subdirectory on local disk.
 *
 * REPORT CONTEXT — ee.Image.getDownloadURL() Parameters:
 * ─────────────────────────────────────────────────────────────────────────────
 * The params object passed to getDownloadURL accepts these key fields:
 *
 *  name    : String prefix for the downloaded file (used in multi-band zips).
 *            For single-image downloads this becomes the .tif filename inside
 *            the returned ZIP. We set it to the point ID for traceability.
 *
 *  bands   : Array of band descriptor objects. Each has:
 *              id          → band name in the source image
 *              min/max     → optional value range for display (not for analysis)
 *            Specifying bands explicitly ensures the download contains exactly
 *            our 8 bands in the correct order, even if the source asset has
 *            additional bands added in a future pipeline iteration.
 *
 *  region  : GeoJSON geometry object defining the spatial extent. We pass the
 *            materialised .bounds() GeoJSON from buildPatchBBox().
 *
 *  scale   : Output pixel size in metres (10 = native Sentinel-2 resolution).
 *
 *  crs     : Output CRS. EPSG:4326 matches all upstream pipeline outputs.
 *
 *  format  : "GEO_TIFF" — single multi-band GeoTIFF (not a ZIP of separate
 *            single-band TIFs). This is the correct value for getDownloadURL.
 *            Note: the correct string is "GEO_TIFF", not "GeoTIFF" or "tiff".
 *
 * IMPORTANT — getDownloadURL returns a ZIP even for GEO_TIFF:
 * ─────────────────────────────────────────────────────────────────────────────
 * Despite specifying format: "GEO_TIFF", GEE wraps the GeoTIFF in a .zip
 * archive when multiple bands are present. The download URL will return a
 * .zip file containing a single .tif. We handle this by:
 *   1. Saving the raw response as a .zip file.
 *   2. Unzipping it in memory using the jszip library.
 *   3. Writing the extracted .tif to the final output path.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {object}    params
 * @param {ee.Image}  params.image       - 8-band feature composite image.
 * @param {object}    params.geojson     - Materialised bounding box GeoJSON.
 * @param {string}    params.pointId     - Used in filename and logs.
 * @param {string}    params.className   - Used to determine output subdirectory.
 * @returns {Promise<{ filepath: string, sizeBytes: number }>}
 */
async function downloadPatch({ image, geojson, pointId, className }) {
  const tag      = `[DOWNLOAD][${pointId}]`;
  const filepath = patchFilePath(className, pointId);

  // ── Step 1: Request a signed download URL from GEE ────────────────────────
  logger.info(`${tag} Requesting GeoTIFF download URL...`);

  const downloadUrl = await new Promise((resolve, reject) => {
    image.getDownloadURL(
      {
        name  : sanitiseName(pointId),
        bands : BAND_NAMES.map((id) => ({ id })),
        region: geojson,
        scale : EXPORT_SCALE_METRES,
        crs   : EXPORT_CRS,
        format: "GEO_TIFF",
      },
      (url, error) => {
        if (error) {
          const errorStr = String(error);

          // Surface the most common GEE getDownloadURL errors with targeted fixes.
          if (errorStr.toLowerCase().includes("total request size")) {
            reject(new Error(
              `${tag} Patch too large for getDownloadURL. ` +
              `Reduce BUFFER_RADIUS_M (currently ${BUFFER_RADIUS_M}m) or ` +
              `reduce the number of bands in BAND_NAMES.`
            ));
          } else if (errorStr.toLowerCase().includes("no data")) {
            reject(new Error(
              `${tag} GEE returned no data for this region. ` +
              `The point may fall outside the feature asset's coverage area. ` +
              `Verify the coordinate against the tile bounding box.`
            ));
          } else {
            reject(new Error(`${tag} getDownloadURL() failed: ${errorStr}`));
          }
        } else if (!url || typeof url !== "string") {
          reject(new Error(
            `${tag} getDownloadURL() returned an empty URL. ` +
            `Ensure the image has valid pixels within the patch bounding box.`
          ));
        } else {
          resolve(url);
        }
      }
    );
  });

  logger.info(`${tag} URL received. Downloading...`);

  // ── Step 2: Download the response ────────────────────────────────────────
  const response = await fetch(downloadUrl);

  if (!response.ok) {
    const isRetryable = response.status >= 500 || response.status === 404;
    throw new Error(
      `${tag} HTTP ${response.status} downloading patch. ` +
      (isRetryable
        ? "Transient server error — will retry."
        : "Non-retryable client error — check band names and region geometry.")
    );
  }

  // ── Step 3: Read response as a buffer to detect ZIP vs raw GeoTIFF ────────
  //
  // REPORT CONTEXT — ZIP handling:
  // ─────────────────────────────────────────────────────────────────────────
  // GEE wraps multi-band GeoTIFF downloads in a ZIP archive. We detect this
  // by reading the first 4 bytes of the response — a ZIP file always starts
  // with the magic bytes 0x50 0x4B 0x03 0x04 ("PK\x03\x04"). If the response
  // is a ZIP, we extract the first .tif entry and write it directly to disk.
  // If it is already a raw GeoTIFF (magic bytes: 0x49 0x49 or 0x4D 0x4D),
  // we write it to disk as-is. This dual handling makes the script robust
  // across GEE API versions that may change the wrapping behaviour.
  // ─────────────────────────────────────────────────────────────────────────
  const arrayBuffer    = await response.arrayBuffer();
  const uint8          = new Uint8Array(arrayBuffer);

  // Check for ZIP magic bytes: PK (0x50, 0x4B)
  const isZip = uint8[0] === 0x50 && uint8[1] === 0x4B;

  let tifBuffer;

  if (isZip) {
    logger.info(`${tag} Response is a ZIP archive. Extracting GeoTIFF...`);

    // Dynamically import jszip — it is a dependency of the pipeline workspace.
    // Dynamic import avoids a hard top-level dependency on environments where
    // jszip may not be installed (e.g. if only the backend workspace runs).
    const { default: JSZip } = await import("jszip");
    const zip = await JSZip.loadAsync(arrayBuffer);

    // Find the first .tif file in the archive.
    const tifEntry = Object.values(zip.files).find(
      (f) => !f.dir && (f.name.endsWith(".tif") || f.name.endsWith(".tiff"))
    );

    if (!tifEntry) {
      throw new Error(
        `${tag} ZIP archive contains no .tif file. ` +
        `Files found: [${Object.keys(zip.files).join(", ")}]. ` +
        `This is an unexpected GEE response format — check GEE API version.`
      );
    }

    logger.info(`${tag} Extracting: ${tifEntry.name}`);
    tifBuffer = await tifEntry.async("nodebuffer");

  } else {
    // Raw GeoTIFF response — write directly.
    logger.info(`${tag} Response is a raw GeoTIFF.`);
    tifBuffer = Buffer.from(arrayBuffer);
  }

  // ── Step 4: Write the GeoTIFF to the class subdirectory ──────────────────
  await fs.writeFile(filepath, tifBuffer);

  // ── Step 5: Verify the file is non-empty ──────────────────────────────────
  const stat = await fs.stat(filepath);
  if (stat.size === 0) {
    await fs.unlink(filepath);
    throw new Error(
      `${tag} Written file is 0 bytes — GEE returned an empty image. ` +
      `Check that the patch bounding box overlaps valid (unmasked) pixels.`
    );
  }

  const sizeKB = (stat.size / 1024).toFixed(1);
  logger.info(`${tag} ✓ Saved: ${filepath} (${sizeKB} KB)`);

  return { filepath, sizeBytes: stat.size };
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 8 — PER-POINT PIPELINE (with retry)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Runs the complete extraction pipeline for a single ground-truth point:
 *   1. Skip if the GeoTIFF already exists on disk.
 *   2. Load the covering feature composite asset.
 *   3. Build the 1280m × 1280m bounding box.
 *   4. Download the GeoTIFF patch to the class subdirectory.
 *
 * @param {object} point    - Flattened ground-truth point descriptor.
 * @param {number} [attempt=1]
 * @returns {Promise<{
 *   pointId  : string,
 *   filepath : string,
 *   sizeBytes: number,
 *   skipped  : boolean
 * }>}
 */
async function processPoint(point, attempt = 1) {
  const { coords, id: pointId, className, notes } = point;
  const [lon, lat] = coords;
  const tag        = `[POINT][${pointId}]`;
  const filepath   = patchFilePath(className, pointId);

  try {
    // ── Step 1: Skip if already downloaded ──────────────────────────────────
    if (await patchFileExists(filepath)) {
      const stat   = await fs.stat(filepath);
      const sizeKB = (stat.size / 1024).toFixed(1);
      logger.info(
        `${tag} Already exists (${sizeKB} KB). Skipping. ` +
        `Delete the file to force re-extraction.`
      );
      return { pointId, filepath, sizeBytes: stat.size, skipped: true };
    }

    // ── Step 2: Load feature composite ──────────────────────────────────────
    const { image } = await loadFeatureCompositeForPoint(lon, lat, pointId);

    // ── Step 3: Build patch bounding box ────────────────────────────────────
    const { geojson } = await buildPatchBBox(lon, lat, pointId);

    // ── Step 4: Download GeoTIFF to class subdirectory ───────────────────────
    const { filepath: savedPath, sizeBytes } = await downloadPatch({
      image,
      geojson,
      pointId,
      className,
    });

    return { pointId, filepath: savedPath, sizeBytes, skipped: false };

  } catch (err) {
    // ── Non-retryable errors ──────────────────────────────────────────────
    const isNonRetryable =
      err.message.includes("is outside all study region")  ||
      err.message.includes("not found")                    ||
      err.message.includes("missing bands")                ||
      err.message.includes("too large")                    ||
      err.message.includes("no data")                      ||
      (err.message.includes("HTTP 4") &&
       !err.message.includes("HTTP 404"));

    if (isNonRetryable) {
      logger.error(`${tag} Non-retryable: ${err.message}`);
      throw err;
    }

    // ── Transient errors — exponential backoff ────────────────────────────
    if (attempt <= MAX_RETRIES) {
      const backoffMs = RETRY_BASE_DELAY_MS * attempt;
      logger.warn(
        `${tag} Transient error (attempt ${attempt}/${MAX_RETRIES}): ${err.message}. ` +
        `Retrying in ${backoffMs / 1000}s...`
      );
      await sleep(backoffMs);
      return processPoint(point, attempt + 1);
    }

    throw new Error(`${tag} Failed after ${MAX_RETRIES} retries: ${err.message}`);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 9 — MAIN ORCHESTRATOR
// ═══════════════════════════════════════════════════════════════════════════════

async function main() {
  logger.info("═".repeat(60));
  logger.info("GALAMSEY SENTINEL — PHASE 3: PATCH EXTRACTION");
  logger.info("═".repeat(60));
  logger.info(`Patches root : ${PATCHES_ROOT}`);
  logger.info(`Features     : ${FEATURES_ASSET_ROOT}`);
  logger.info(`Data year    : ${DATA_YEAR}`);
  logger.info(`Patch size   : ~${(BUFFER_RADIUS_M * 2) / EXPORT_SCALE_METRES}×${(BUFFER_RADIUS_M * 2) / EXPORT_SCALE_METRES}px ` +
              `(${BUFFER_RADIUS_M * 2}m × ${BUFFER_RADIUS_M * 2}m @ ${EXPORT_SCALE_METRES}m)`);
  logger.info(`Bands        : [${BAND_NAMES.join(", ")}]`);

  // ── 1. Authenticate ───────────────────────────────────────────────────────
  try {
    await authenticateGEE();
  } catch (authError) {
    logger.error(`Fatal: GEE authentication failed. ${authError.message}`);
    process.exit(1);
  }

  // ── 2. Create output directory structure ─────────────────────────────────
  try {
    await ensureOutputDirectories();
  } catch (dirError) {
    logger.error(`Fatal: Cannot create output directories. ${dirError.message}`);
    process.exit(1);
  }

  // ── 3. Flatten and validate ground-truth points ───────────────────────────
  const allPoints = flattenGroundTruthPoints();

  if (allPoints.length === 0) {
    logger.error(
      "No ground-truth points defined. " +
      "Add coordinates to GROUND_TRUTH_POINTS in Section 1 before running."
    );
    process.exit(1);
  }

  // Print pre-run class breakdown.
  const classCounts = allPoints.reduce((acc, p) => {
    acc[p.className] = (acc[p.className] ?? 0) + 1;
    return acc;
  }, {});

  logger.info(`\nTotal points: ${allPoints.length}`);
  Object.entries(classCounts).forEach(([cls, n]) => {
    logger.info(`  Class ${CLASS_LABELS[cls]} (${cls}): ${n} point(s) → ${classDirName(cls)}/`);
  });
  logger.info("─".repeat(60));

  // ── 4. Process each point sequentially ───────────────────────────────────
  const results = [];
  const errors  = [];

  for (let i = 0; i < allPoints.length; i++) {
    const point = allPoints[i];

    logger.info(
      `\n── Point ${i + 1}/${allPoints.length}: ${point.id} ` +
      `(${point.className}, label=${point.classLabel}) ──`
    );
    logger.info(`   Coordinates : [${point.coords.join(", ")}]`);
    if (point.notes) logger.info(`   Notes       : ${point.notes}`);

    try {
      const result = await processPoint(point);
      results.push(result);
    } catch (pointError) {
      logger.error(`[${point.id}] ${pointError.message}`);
      errors.push({ pointId: point.id, error: pointError.message });
    }

    if (i < allPoints.length - 1) {
      logger.info(`Waiting ${INTER_POINT_DELAY_MS / 1000}s...`);
      await sleep(INTER_POINT_DELAY_MS);
    }
  }

  // ── 5. Final manifest ─────────────────────────────────────────────────────
  const downloaded = results.filter((r) => !r.skipped);
  const skipped    = results.filter((r) =>  r.skipped);
  const totalBytes = downloaded.reduce((s, r) => s + r.sizeBytes, 0);

  logger.info(`\n${"═".repeat(60)}`);
  logger.info("PHASE 3 — PATCH EXTRACTION MANIFEST");
  logger.info(`${"═".repeat(60)}`);
  logger.info(`  Output root   : ${PATCHES_ROOT}`);
  logger.info(`  Total points  : ${allPoints.length}`);
  logger.info(`  ✓ Downloaded  : ${downloaded.length}  (${(totalBytes / 1024).toFixed(0)} KB total)`);
  logger.info(`  ⚠ Skipped     : ${skipped.length}  (file already existed)`);
  logger.info(`  ✗ Failed      : ${errors.length}`);
  logger.info(`${"═".repeat(60)}`);
  logger.info(`  Folder structure:`);

  for (const className of Object.keys(CLASS_LABELS)) {
    const classResults = results.filter((r) => !r.skipped &&
      r.filepath.includes(classDirName(className)));
    const classSkipped = skipped.filter((r) =>
      r.filepath.includes(classDirName(className)));
    logger.info(`\n  ${classDirName(className)}/`);
    [...classResults, ...classSkipped].forEach((r) => {
      const sizeKB = (r.sizeBytes / 1024).toFixed(1);
      const flag   = r.skipped ? "⚠" : "✓";
      logger.info(`    ${flag} ${path.basename(r.filepath)}  (${sizeKB} KB)`);
    });
  }

  if (errors.length > 0) {
    logger.info(`\n  Failures:`);
    errors.forEach((e) => logger.warn(`    ✗ ${e.pointId} — ${e.error}`));
  }

  // PyTorch / TF loading snippets for Phase 4.
  logger.info(`\n${"─".repeat(60)}`);
  logger.info("  PyTorch ImageFolder loading snippet:");
  logger.info(`${"─".repeat(60)}`);
  logger.info(`  from torchvision import datasets, transforms`);
  logger.info(`  dataset = datasets.ImageFolder(`);
  logger.info(`      root="${PATCHES_ROOT}",`);
  logger.info(`      transform=transforms.ToTensor()`);
  logger.info(`  )`);
  logger.info(`  # Classes auto-detected: ${Object.keys(CLASS_LABELS).map((c) => classDirName(c)).join(", ")}`);
  logger.info(`\n  TensorFlow/Keras loading snippet:`);
  logger.info(`  dataset = tf.keras.utils.image_dataset_from_directory(`);
  logger.info(`      "${PATCHES_ROOT}",`);
  logger.info(`      image_size=(${BUFFER_RADIUS_M * 2 / EXPORT_SCALE_METRES}, ${BUFFER_RADIUS_M * 2 / EXPORT_SCALE_METRES}),`);
  logger.info(`      batch_size=16`);
  logger.info(`  )`);
  logger.info(`${"═".repeat(60)}\n`);

  if (errors.length > 0 && downloaded.length === 0 && skipped.length === 0) {
    logger.error("All points failed. Exiting with error code.");
    process.exit(1);
  }
}

// ─── Exports ──────────────────────────────────────────────────────────────────
export {
  GROUND_TRUTH_POINTS,
  CLASS_LABELS,
  BAND_NAMES,
  BUFFER_RADIUS_M,
  EXPORT_SCALE_METRES,
  DATA_YEAR,
  PATCHES_ROOT,
  FEATURES_ASSET_ROOT,
  STUDY_REGIONS,
  flattenGroundTruthPoints,
  buildPatchBBox,
  downloadPatch,
  loadFeatureCompositeForPoint,
  processPoint,
  ensureOutputDirectories,
  patchFileExists,
  classDirName,
  classDirPath,
  patchFilePath,
  generateTiles,
  sanitiseName,
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