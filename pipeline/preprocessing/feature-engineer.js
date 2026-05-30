/**
 * @file feature-engineer.js
 * @description Phase 2.5 — Feature Engineering Pipeline.
 *              Loads exported Sentinel-1 and Sentinel-2 Earth Engine Assets,
 *              computes spectral indices (NDVI, NDWI), stacks all layers into
 *              an 8-band master composite, and exports each tile to the
 *              processed_features asset folder for CNN training.
 *
 * OUTPUT BAND STACK (per tile):
 *  ┌────────┬──────┬────────────────────────────────────────────────────────┐
 *  │ Index  │ Band │ Description                                            │
 *  ├────────┼──────┼────────────────────────────────────────────────────────┤
 *  │   0    │  B2  │ S2 Blue  — water clarity, visual composite             │
 *  │   1    │  B3  │ S2 Green — visual composite, NDWI numerator            │
 *  │   2    │  B4  │ S2 Red   — visual composite, NDVI denominator term     │
 *  │   3    │  B8  │ S2 NIR   — vegetation, NDVI/NDWI denominator term      │
 *  │   4    │  VV  │ S1 SAR co-pol  — surface roughness, bare soil          │
 *  │   5    │  VH  │ S1 SAR cross-pol — vegetation volume scattering        │
 *  │   6    │ NDVI │ Normalised Difference Vegetation Index                 │
 *  │   7    │ NDWI │ Normalised Difference Water Index                      │
 *  └────────┴──────┴────────────────────────────────────────────────────────┘
 *
 * REPORT CONTEXT — Why Feature Engineering Before the CNN?
 * ─────────────────────────────────────────────────────────────────────────────
 * Raw satellite bands are powerful but redundant. The CNN must learn to
 * distinguish three target classes from pixel values:
 *
 *   Class 0 — Dense Forest / Intact Vegetation
 *   Class 1 — Active Galamsey Mining Pit (bare soil, excavation)
 *   Class 2 — Water Body / River Surface
 *
 * While a deep CNN could theoretically derive the optimal features from raw
 * bands alone, providing pre-computed spectral indices as explicit channels
 * offers three concrete benefits:
 *
 *  1. FASTER CONVERGENCE: The network receives a direct representation of
 *     vegetation health (NDVI) and water extent (NDWI), reducing the number
 *     of layers and epochs needed to learn the same discriminative features
 *     from raw band ratios.
 *
 *  2. PHYSICS-GROUNDED FEATURES: NDVI and NDWI are normalised ratios that
 *     are inherently invariant to scene brightness — they respond to the
 *     spectral shape of the surface rather than its absolute reflectance.
 *     This makes them more robust to illumination variation across tiles
 *     collected at different times of year.
 *
 *  3. MULTI-SENSOR FUSION: Combining S2 optical bands with S1 SAR bands in
 *     a single tensor is the core fusion strategy of this project. The 8-band
 *     stack means every training patch carries optical texture (B2–B8),
 *     radar backscatter (VV, VH), vegetation health (NDVI), and water
 *     surface extent (NDWI) simultaneously — a richer representation than
 *     either sensor alone.
 *
 * REPORT CONTEXT — NDVI Formula:
 * ─────────────────────────────────────────────────────────────────────────────
 * NDVI = (NIR − Red) / (NIR + Red) = (B8 − B4) / (B8 + B4)
 *
 * Range: −1 to +1
 *   > 0.5  → Dense, healthy vegetation (intact forest)
 *   0.2–0.5 → Sparse vegetation, degraded land
 *   0–0.2  → Bare soil, exposed rock, mining pits
 *   < 0    → Water, cloud shadow, saturated soil
 *
 * Mining pits produce characteristically low NDVI (0–0.15) because
 * excavation removes the vegetation canopy entirely, exposing mineral
 * soil with high red reflectance and low NIR reflectance.
 *
 * REPORT CONTEXT — NDWI Formula:
 * ─────────────────────────────────────────────────────────────────────────────
 * NDWI = (Green − NIR) / (Green + NIR) = (B3 − B8) / (B3 + B8)
 *                                        [McFeeters 1996]
 *
 * Range: −1 to +1
 *   > 0    → Open water (rivers, turbid mining ponds)
 *   0–0.2  → Soil moisture, saturated ground near rivers
 *   < 0    → Vegetation, bare soil, urban surfaces
 *
 * In our Galamsey context, NDWI is dual-purpose: it delineates the major
 * river channels (Pra, Ankobra, Birim) where we measure turbidity, AND it
 * highlights the water-filled pits and sluice ponds characteristic of
 * alluvial gold mining operations.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @module feature-engineer
 */

import ee from "@google/earthengine";
import path from "path";
import { fileURLToPath } from "url";
import "dotenv/config";
import { authenticateGEE, logger } from "../ingestion/gee-auth.js";

// ─── ES Module __dirname Shim ─────────────────────────────────────────────────
const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// ─── Source Asset Paths ───────────────────────────────────────────────────────

/**
 * Root path of the exported Sentinel-2 optical tiles (Phase 1 output).
 * Each tile asset is addressable as: S2_ASSET_ROOT/<tileName>_<year>
 */
const S2_ASSET_ROOT = process.env.GEE_ASSET_ROOT ||
  "projects/galamsey-sentinel/assets/sentinel2_tiles";

/**
 * Root path of the exported Sentinel-1 SAR tiles (Phase 1 output).
 * Each tile asset is addressable as: S1_ASSET_ROOT/<tileName>_S1_<year>
 */
const S1_ASSET_ROOT = process.env.GEE_S1_ASSET_ROOT ||
  "projects/galamsey-sentinel/assets/sentinel1_tiles";

// ─── Output Asset Path ────────────────────────────────────────────────────────

/**
 * Destination asset folder for the 8-band processed feature stacks.
 *
 * REPORT CONTEXT — Folder Separation by Pipeline Stage:
 * ─────────────────────────────────────────────────────────────────────────────
 * Keeping raw sensor tiles (sentinel1_tiles, sentinel2_tiles) separate from
 * engineered features (processed_features) is a deliberate data management
 * decision. It means:
 *
 *  1. Raw tiles can be re-used for other experiments (e.g. turbidity analysis
 *     in Phase 3) without being polluted by derived products.
 *  2. The feature engineering step can be re-run with different index
 *     combinations (e.g. adding EVI or MNDWI) without touching Phase 1.
 *  3. The CNN training script (Phase 3) has a single, unambiguous data source.
 * ─────────────────────────────────────────────────────────────────────────────
 */
const OUTPUT_ASSET_ROOT = process.env.GEE_FEATURES_ASSET_ROOT ||
  "projects/galamsey-sentinel/assets/processed_features";

// ─── Band & Index Constants ───────────────────────────────────────────────────

/** S2 bands required for index computation and the final stack. */
const S2_BANDS = ["B2", "B3", "B4", "B8"];

/** S1 SAR polarisation bands included in the final stack. */
const S1_BANDS = ["VV", "VH"];

/** Ordered band names of the final 8-channel output asset. */
const OUTPUT_BAND_ORDER = ["B2", "B3", "B4", "B8", "VV", "VH", "NDVI", "NDWI"];

/** Data year — must match the year suffix used in sentinel1/2-fetcher.js. */
const DATA_YEAR = parseInt(process.env.DATA_YEAR) || 2025;

// ─── Processing Constants ─────────────────────────────────────────────────────

/** Maximum tile area in km² — identical to fetcher scripts for grid alignment. */
const MAX_TILE_AREA_KM2 = 2000;

/** Milliseconds to wait between sequential tile processing runs. */
const INTER_TILE_DELAY_MS = 3000;

/** Maximum retry attempts per tile for transient GEE API errors. */
const MAX_RETRIES = 3;

/** Base delay in ms for exponential backoff on retries. */
const RETRY_BASE_DELAY_MS = 10_000;

// ─── Export Constants ─────────────────────────────────────────────────────────

/**
 * Export pixel size in metres.
 * Must match the fetcher scripts (10m) to maintain pixel-level alignment
 * between source tiles and the processed feature stack.
 */
const EXPORT_SCALE_METRES = 10;

/** CRS for all exports. WGS84 matches both fetcher outputs. */
const EXPORT_CRS = "EPSG:4326";

/** Polling interval in ms between ee.data.getTaskStatus() calls. */
const POLL_INTERVAL_MS = 15_000;

/** Maximum ms to wait for a single export task before timing out (2 hours). */
const EXPORT_TIMEOUT_MS =
  parseInt(process.env.GEE_EXPORT_TIMEOUT_MS) || 7_200_000;

// ─── Study Regions ────────────────────────────────────────────────────────────
/**
 * Identical bounding boxes to both fetcher scripts.
 * Grid alignment is only guaranteed if these coordinates are unchanged.
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
 * @param {number[]} bbox - [west, south, east, north]
 * @returns {number} Estimated area in km²
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
 * @param {string} name
 * @returns {string} GEE-safe asset name
 */
function sanitiseAssetName(name) {
  return name
    .replace(/[^a-zA-Z0-9_-]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "");
}

// ─── Utility: BBox Validator ──────────────────────────────────────────────────
/**
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
 * Reproduces the exact tile grid from the fetcher scripts.
 * Deterministic: same bbox + same MAX_TILE_AREA_KM2 → same tile IDs.
 *
 * @param {number[]} bbox
 * @param {string}   regionName
 * @returns {Array<{ tileId: string, bbox: number[], row: number, col: number }>}
 */
function generateTiles(bbox, regionName) {
  const [west, south, east, north] = bbox;
  const totalArea = estimateAreaKm2(bbox);

  if (totalArea <= MAX_TILE_AREA_KM2) {
    return [{ tileId: `${regionName}_tile_r0_c0`, bbox, row: 0, col: 0 }];
  }

  const tilesNeeded = Math.ceil(totalArea / MAX_TILE_AREA_KM2);
  const gridSize    = Math.ceil(Math.sqrt(tilesNeeded));
  const rows        = gridSize;
  const cols        = gridSize;
  const tileWidth   = (east  - west)  / cols;
  const tileHeight  = (north - south) / rows;

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

// ─── Asset Helpers ────────────────────────────────────────────────────────────

/**
 * Ensures an EE asset folder exists, creating it if absent.
 * @param {string} assetPath
 * @returns {Promise<void>}
 */
async function createAssetFolderIfMissing(assetPath) {
  logger.info(`[ASSET-SETUP] Verifying asset folder: ${assetPath}`);
  await new Promise((resolve, reject) => {
    ee.data.createFolder(assetPath, false, (result, error) => {
      if (!error) {
        logger.info(`[ASSET-SETUP] ✓ Created: ${assetPath}`);
        return resolve();
      }
      const e = String(error).toLowerCase();
      if (e.includes("already exists") || e.includes("cannot overwrite")) {
        logger.info(`[ASSET-SETUP] ✓ Already exists: ${assetPath}`);
        return resolve();
      }
      reject(new Error(
        `[ASSET-SETUP] Failed to create "${assetPath}": ${error}. ` +
        `Ensure the Cloud Project ID matches your registered GEE project.`
      ));
    });
  });
}

/**
 * Returns true if an EE asset exists at the given path.
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

// ─── Asset Name Resolvers ─────────────────────────────────────────────────────

/**
 * Reconstructs the S2 asset ID for a given tile using the same naming
 * convention as sentinel2-fetcher.js.
 *
 * sentinel2-fetcher used: sanitiseAssetName(`${tileId}_${year}`)
 * Example: "Pra_River_Basin__Western_Ashanti_Region__tile_r0_c0_2025"
 *
 * @param {string} tileId
 * @returns {string} Full EE asset path.
 */
function resolveS2AssetId(tileId) {
  const name = sanitiseAssetName(`${tileId}_${DATA_YEAR}`);
  return `${S2_ASSET_ROOT}/${name}`;
}

/**
 * Reconstructs the S1 asset ID for a given tile using the same naming
 * convention as sentinel1-fetcher.js.
 *
 * sentinel1-fetcher used: sanitiseAssetName(`${tileId}_S1_${year}`)
 * Example: "Pra_River_Basin__Western_Ashanti_Region__tile_r0_c0_S1_2025"
 *
 * @param {string} tileId
 * @returns {string} Full EE asset path.
 */
function resolveS1AssetId(tileId) {
  const name = sanitiseAssetName(`${tileId}_S1_${DATA_YEAR}`);
  return `${S1_ASSET_ROOT}/${name}`;
}

/**
 * Constructs the output asset ID for the processed feature stack.
 *
 * @param {string} tileId
 * @returns {string} Full EE asset path.
 */
function resolveOutputAssetId(tileId) {
  const name = sanitiseAssetName(`${tileId}_features_${DATA_YEAR}`);
  return `${OUTPUT_ASSET_ROOT}/${name}`;
}

// ─── Core Feature Engineering ─────────────────────────────────────────────────

/**
 * Loads, validates, and verifies band availability for a source EE asset.
 *
 * REPORT CONTEXT — Why explicit band validation?
 * ─────────────────────────────────────────────────────────────────────────────
 * GEE's lazy evaluation model means that loading an asset with ee.Image() never
 * fails — it only fails when the image is actually computed (during export or
 * .evaluate()). If a required band is missing, the export task fails silently
 * minutes later with a cryptic error. By calling .bandNames().evaluate() here,
 * we trigger an early, synchronous check that surfaces missing bands immediately
 * with a clear, actionable error message before any export is submitted.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {string}   assetId       - Full EE asset path to load.
 * @param {string[]} requiredBands - Band names that must be present.
 * @param {string}   tag           - Log prefix for error messages.
 * @returns {Promise<ee.Image>}    - The loaded image, validated.
 * @throws {Error} If the asset is missing or required bands are absent.
 */
async function loadAndValidateAsset(assetId, requiredBands, tag) {
  // Confirm the asset exists before attempting to load it.
  const exists = await assetExists(assetId);
  if (!exists) {
    throw new Error(
      `${tag} Source asset not found: "${assetId}". ` +
      `Ensure Phase 1 fetcher scripts completed successfully for this tile.`
    );
  }

  // Load the asset into a server-side ee.Image object.
  // This is instantaneous — no pixel data is transferred.
  const image = ee.Image(assetId);

  // Trigger a lightweight server-side call to retrieve band names.
  // This is the earliest point at which we can detect missing bands.
  const bandNames = await new Promise((resolve, reject) => {
    image.bandNames().evaluate((result, error) => {
      if (error) {
        reject(new Error(`${tag} Failed to read band names from "${assetId}": ${error}`));
      } else {
        resolve(result);
      }
    });
  });

  // Check that every required band is present in the loaded asset.
  const missingBands = requiredBands.filter((b) => !bandNames.includes(b));
  if (missingBands.length > 0) {
    throw new Error(
      `${tag} Asset "${assetId}" is missing required bands: ` +
      `[${missingBands.join(", ")}]. ` +
      `Found: [${bandNames.join(", ")}]. ` +
      `Re-run the Phase 1 fetcher for this tile.`
    );
  }

  logger.info(
    `${tag} Loaded asset: ${assetId} | Bands confirmed: [${bandNames.join(", ")}]`
  );

  return image;
}

/**
 * Computes NDVI from a Sentinel-2 image.
 *
 * REPORT CONTEXT — ee.Image.normalizedDifference():
 * ─────────────────────────────────────────────────────────────────────────────
 * GEE provides a built-in normalizedDifference(bandA, bandB) method that
 * computes (bandA − bandB) / (bandA + bandB) with automatic handling of
 * division-by-zero (outputs 0 where the denominator is 0). This is more
 * numerically stable than manually computing the ratio with .subtract()
 * and .divide(), as it avoids NaN or Infinity values in flat-response pixels.
 *
 * The output is a single-band image named "nd" by default.
 * We rename it to "NDVI" immediately for clarity in the stacked output.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {ee.Image} s2Image - Sentinel-2 image with B4 and B8 bands.
 * @returns {ee.Image} Single-band image named "NDVI", range [−1, +1].
 */
function computeNDVI(s2Image) {
  return s2Image
    .normalizedDifference(["B8", "B4"]) // (NIR − Red) / (NIR + Red)
    .rename("NDVI");
}

/**
 * Computes NDWI from a Sentinel-2 image.
 *
 * REPORT CONTEXT — McFeeters (1996) NDWI vs Gao (1996) NDWI:
 * ─────────────────────────────────────────────────────────────────────────────
 * Two different indices share the acronym NDWI:
 *
 *  • McFeeters (1996): (Green − NIR) / (Green + NIR) = (B3 − B8) / (B3 + B8)
 *    → Designed to detect OPEN WATER SURFACES. Positive values = water.
 *    → This is the version we use — appropriate for river delineation and
 *      mining pond detection.
 *
 *  • Gao (1996): (NIR − SWIR) / (NIR + SWIR)
 *    → Designed to detect VEGETATION WATER CONTENT (leaf moisture).
 *    → Requires a SWIR band (B11 or B12) which is at 20m resolution,
 *      requiring resampling that would degrade our 10m stack.
 *
 * We explicitly document this choice because both are called "NDWI" in
 * the literature and the distinction matters for result interpretation.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {ee.Image} s2Image - Sentinel-2 image with B3 and B8 bands.
 * @returns {ee.Image} Single-band image named "NDWI", range [−1, +1].
 */
function computeNDWI(s2Image) {
  return s2Image
    .normalizedDifference(["B3", "B8"]) // (Green − NIR) / (Green + NIR)
    .rename("NDWI");
}

/**
 * Builds the 8-band master feature stack for a single tile.
 *
 * REPORT CONTEXT — ee.Image.addBands() Stacking Strategy:
 * ─────────────────────────────────────────────────────────────────────────────
 * ee.Image.addBands() appends new band(s) to an existing image. Since all
 * source layers (S2, S1, NDVI, NDWI) share the same spatial extent, CRS,
 * and scale (EPSG:4326 at 10m), no reprojection occurs — this is a pure
 * metadata-level band concatenation that GEE resolves lazily at export time.
 *
 * The explicit .select(OUTPUT_BAND_ORDER) call at the end enforces a
 * guaranteed band ordering regardless of the order in which addBands()
 * was called. This is critical for the CNN: PyTorch/TensorFlow.js index
 * bands positionally, so band 0 must always be B2 and band 6 must always
 * be NDVI in every tile across every run.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {object}   params
 * @param {ee.Image} params.s2Image   - Validated S2 asset with [B2,B3,B4,B8].
 * @param {ee.Image} params.s1Image   - Validated S1 asset with [VV,VH].
 * @param {string}   params.tileId    - For logging.
 * @returns {ee.Image} 8-band image with bands in OUTPUT_BAND_ORDER.
 */
function buildFeatureStack({ s2Image, s1Image, tileId }) {
  const tag = `[STACK][${tileId}]`;

  // ── Compute spectral indices ──────────────────────────────────────────────
  const ndvi = computeNDVI(s2Image);
  const ndwi = computeNDWI(s2Image);

  logger.info(`${tag} NDVI and NDWI computed.`);

  // ── Stack all 8 bands into a single image ─────────────────────────────────
  // Starting from the S2 image [B2, B3, B4, B8], we progressively add:
  //   + S1 bands  → [B2, B3, B4, B8, VV, VH]
  //   + NDVI      → [B2, B3, B4, B8, VV, VH, NDVI]
  //   + NDWI      → [B2, B3, B4, B8, VV, VH, NDVI, NDWI]
  const stackedImage = s2Image
    .addBands(s1Image.select(S1_BANDS))
    .addBands(ndvi)
    .addBands(ndwi)
    .select(OUTPUT_BAND_ORDER); // enforce guaranteed band ordering

  logger.info(
    `${tag} Feature stack built. ` +
    `Band order: [${OUTPUT_BAND_ORDER.join(", ")}]`
  );

  return stackedImage;
}

// ─── Per-Tile Processing Pipeline ────────────────────────────────────────────

/**
 * Executes the full feature engineering pipeline for a single tile:
 *   1. Resolve source asset IDs from the tile ID.
 *   2. Load and validate both source assets.
 *   3. Build the 8-band feature stack.
 *   4. Return the stacked image + metadata.
 *
 * Includes exponential backoff retry for transient GEE API errors.
 *
 * @param {object} params
 * @param {string}   params.tileId
 * @param {number[]} params.bbox
 * @param {number}   [params.attempt=1]
 *
 * @returns {Promise<{
 *   image    : ee.Image,
 *   region   : ee.Geometry,
 *   metadata : object
 * } | null>} Null if either source asset is missing.
 */
async function processTile({ tileId, bbox, attempt = 1 }) {
  const tag = `[PROCESS][${tileId}]`;

  try {
    // ── 1. Resolve the source asset IDs ──────────────────────────────────────
    const s2AssetId = resolveS2AssetId(tileId);
    const s1AssetId = resolveS1AssetId(tileId);

    logger.info(`${tag} Source S2 asset : ${s2AssetId}`);
    logger.info(`${tag} Source S1 asset : ${s1AssetId}`);

    // ── 2. Load and validate both source assets ───────────────────────────────
    // We run these sequentially rather than in parallel (Promise.all) to avoid
    // hitting GEE's concurrent request limit on the free tier.
    const s2Image = await loadAndValidateAsset(s2AssetId, S2_BANDS, tag);
    const s1Image = await loadAndValidateAsset(s1AssetId, S1_BANDS, tag);

    // ── 3. Build the tile's geometry for export clipping ─────────────────────
    const [west, south, east, north] = bbox;
    const region = ee.Geometry.Rectangle([west, south, east, north], null, false);

    // ── 4. Build the 8-band feature stack ────────────────────────────────────
    const featureStack = buildFeatureStack({ s2Image, s1Image, tileId });

    // ── 5. Verify output band names via a lightweight server-side call ────────
    const outputBands = await new Promise((resolve, reject) => {
      featureStack.bandNames().evaluate((result, error) => {
        if (error) reject(new Error(`${tag} Output band validation failed: ${error}`));
        else resolve(result);
      });
    });

    // Confirm the output matches our expected 8-band specification exactly.
    const bandMismatch = OUTPUT_BAND_ORDER.some((b, i) => b !== outputBands[i]);
    if (outputBands.length !== OUTPUT_BAND_ORDER.length || bandMismatch) {
      throw new Error(
        `${tag} Output band order mismatch. ` +
        `Expected: [${OUTPUT_BAND_ORDER.join(", ")}]. ` +
        `Got: [${outputBands.join(", ")}].`
      );
    }

    logger.info(
      `${tag} ✓ Output validated. 8-band stack: [${outputBands.join(", ")}]`
    );

    const metadata = {
      tileId,
      bbox,
      s2AssetId,
      s1AssetId,
      outputBands,
      dataYear     : DATA_YEAR,
      ndviFormula  : "(B8 - B4) / (B8 + B4)",
      ndwiFormula  : "(B3 - B8) / (B3 + B8)",
      ndwiReference: "McFeeters (1996)",
      createdAt    : new Date().toISOString(),
    };

    return { image: featureStack, region, metadata };

  } catch (err) {
    // ── Missing-asset errors are not retryable ────────────────────────────────
    // If the source asset doesn't exist, no amount of retrying will create it.
    // Return null so the orchestrator can log a clear skip message and continue.
    if (
      err.message.includes("not found") ||
      err.message.includes("missing required bands")
    ) {
      logger.warn(`${tag} Non-retryable error: ${err.message}`);
      return null;
    }

    // ── Transient GEE errors — exponential backoff retry ─────────────────────
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
      return processTile({ tileId, bbox, attempt: attempt + 1 });
    }

    throw new Error(`${tag} Failed after ${attempt - 1} retries: ${err.message}`);
  }
}

// ─── Export: Submit toAsset Task ──────────────────────────────────────────────

/**
 * Submits a GEE toAsset export task for a processed feature stack tile.
 *
 * REPORT CONTEXT — Pyramiding Policy for Mixed-Type Bands:
 * ─────────────────────────────────────────────────────────────────────────────
 * Our 8-band stack contains two types of continuous data:
 *
 *  • Reflectance / backscatter (B2–B8, VV, VH): physical energy measurements.
 *    "MEAN" averaging is radiometrically correct for these bands at lower zoom.
 *
 *  • Normalised indices (NDVI, NDWI): ratio values in [−1, +1]. "MEAN" is also
 *    correct here — the mean of a set of NDVI values is a valid representative
 *    NDVI for that area, unlike categorical data where "MEAN" would be wrong.
 *
 * All 8 bands therefore use "MEAN" pyramiding — no exceptions needed.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {object}      params
 * @param {ee.Image}    params.image
 * @param {ee.Geometry} params.region
 * @param {object}      params.metadata
 * @returns {Promise<{ taskId: string, assetId: string, status: "SUBMITTED" }>}
 */
async function submitExportTask({ image, region, metadata }) {
  const tag = `[EXPORT][${metadata.tileId}]`;

  const assetId = resolveOutputAssetId(metadata.tileId);

  logger.info(`${tag} Submitting toAsset export.`);
  logger.info(`${tag} Destination: ${assetId}`);
  logger.info(
    `${tag} Bands: [${metadata.outputBands.join(", ")}] | ` +
    `Scale: ${EXPORT_SCALE_METRES}m | CRS: ${EXPORT_CRS}`
  );

  // Build per-band pyramiding policy object from OUTPUT_BAND_ORDER.
  // Generating it programmatically ensures it stays in sync if bands are
  // added or reordered in a future iteration.
  const pyramidingPolicy = OUTPUT_BAND_ORDER.reduce((policy, band) => {
    policy[band] = "MEAN";
    return policy;
  }, {});

  const assetName = assetId.split("/").pop();

  const task = ee.batch.Export.image.toAsset({
    image,
    description     : assetName,
    assetId         : assetId,
    scale           : EXPORT_SCALE_METRES,
    region          : region,
    crs             : EXPORT_CRS,
    maxPixels       : 1e9,
    pyramidingPolicy: pyramidingPolicy,
  });

  await new Promise((resolve, reject) => {
    try {
      task.start();
      resolve();
    } catch (startError) {
      reject(new Error(
        `${tag} task.start() failed: ${startError.message}. ` +
        `Common causes: (1) Output folder does not exist. ` +
        `(2) Asset ID "${assetId}" already exists. ` +
        `(3) GEE auth token expired.`
      ));
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
 * Polls a GEE export task until it reaches a terminal state or times out.
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
      logger.warn(`${tag} Status check error: ${pollError.message}. Retrying...`);
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

// ─── Region Orchestrator ──────────────────────────────────────────────────────

/**
 * Runs the feature engineering pipeline for all tiles in a single region.
 *
 * Per-tile order:
 *   1. Skip if output asset already exists  (resumable).
 *   2. Process tile (load → validate → stack) → processTile().
 *   3. Submit toAsset export task           → submitExportTask().
 *   4. Poll until terminal state            → pollExportTask().
 *   5. Wait INTER_TILE_DELAY_MS before next tile.
 *
 * @param {object}   params
 * @param {number[]} params.bbox
 * @param {string}   params.regionName
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
async function processRegion({ bbox, regionName }) {
  logger.info(`\n${"─".repeat(60)}`);
  logger.info(`[${regionName}] Starting feature engineering.`);

  const validation = validateBoundingBox(bbox);
  if (!validation.isValid) {
    throw new Error(`[${regionName}] Invalid bounding box: ${validation.reason}`);
  }

  const tiles = generateTiles(bbox, regionName);
  logger.info(`[${regionName}] ${tiles.length} tile(s) to process.`);

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
      // ── Step 1: Skip if output asset already exists ───────────────────────
      const outputAssetId = resolveOutputAssetId(tileId);
      const alreadyDone   = await assetExists(outputAssetId);

      if (alreadyDone) {
        logger.info(
          `[${tileId}] Output already exists at ${outputAssetId}. Skipping.`
        );
        skippedCount++;
        if (i < tiles.length - 1) await sleep(INTER_TILE_DELAY_MS);
        continue;
      }

      // ── Step 2: Process tile (load + validate + stack) ────────────────────
      const processResult = await processTile({ tileId, bbox: tileBbox });

      if (processResult === null) {
        // Source asset(s) missing — non-retryable. Count as skipped.
        skippedCount++;
        logger.warn(
          `[${tileId}] Skipped — source asset(s) missing. ` +
          `Re-run Phase 1 fetcher scripts for this tile.`
        );
        if (i < tiles.length - 1) await sleep(INTER_TILE_DELAY_MS);
        continue;
      }

      const { image, region, metadata } = processResult;

      // ── Step 3: Submit toAsset export task ────────────────────────────────
      const submission = await submitExportTask({ image, region, metadata });

      // ── Step 4: Poll until terminal state ────────────────────────────────
      const pollResult = await pollExportTask(submission.taskId, submission.assetId);

      exportResults.push({
        tileId,
        ...submission,
        ...pollResult,
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
 * Entry point. Authenticates with GEE, ensures the output asset folder exists,
 * then runs the feature engineering pipeline for all study regions.
 */
async function main() {
  logger.info("═".repeat(60));
  logger.info("GALAMSEY SENTINEL — PHASE 2.5: FEATURE ENGINEERING");
  logger.info("═".repeat(60));
  logger.info(`Data year    : ${DATA_YEAR}`);
  logger.info(`S2 source    : ${S2_ASSET_ROOT}`);
  logger.info(`S1 source    : ${S1_ASSET_ROOT}`);
  logger.info(`Output       : ${OUTPUT_ASSET_ROOT}`);
  logger.info(`Band stack   : [${OUTPUT_BAND_ORDER.join(", ")}]`);
  logger.info(`Export scale : ${EXPORT_SCALE_METRES}m | CRS: ${EXPORT_CRS}`);
  logger.info("─".repeat(60));

  // ── 1. Authenticate ───────────────────────────────────────────────────────
  try {
    await authenticateGEE();
  } catch (authError) {
    logger.error(`Fatal: GEE authentication failed. ${authError.message}`);
    process.exit(1);
  }

  // ── 2. Ensure the output asset folder exists ──────────────────────────────
  try {
    await createAssetFolderIfMissing(OUTPUT_ASSET_ROOT);
  } catch (folderError) {
    logger.error(`Fatal: Output folder setup failed. ${folderError.message}`);
    process.exit(1);
  }

  // ── 3. Process each study region ──────────────────────────────────────────
  const allResults = [];
  const allErrors  = [];

  for (const [key, region] of Object.entries(STUDY_REGIONS)) {
    try {
      const result = await processRegion({
        bbox      : region.bbox,
        regionName: region.name,
      });
      allResults.push({ regionKey: key, ...result });
    } catch (regionError) {
      logger.error(`Fatal error for "${region.name}": ${regionError.message}`);
      allErrors.push({ regionKey: key, error: regionError.message });
    }
  }

  // ── 4. Final manifest ─────────────────────────────────────────────────────
  const totalTiles    = allResults.reduce((s, r) => s + r.tileCount,    0);
  const totalExported = allResults.reduce((s, r) => s + r.successCount, 0);
  const totalSkipped  = allResults.reduce((s, r) => s + r.skippedCount, 0);
  const totalFailed   = allResults.reduce((s, r) => s + r.failedCount,  0);

  logger.info(`\n${"═".repeat(60)}`);
  logger.info("PHASE 2.5 — FEATURE ENGINEERING MANIFEST");
  logger.info(`${"═".repeat(60)}`);
  logger.info(`  Output folder : ${OUTPUT_ASSET_ROOT}`);
  logger.info(`  Band stack    : [${OUTPUT_BAND_ORDER.join(", ")}]`);
  logger.info(`  Data year     : ${DATA_YEAR}`);
  logger.info(`  Total tiles   : ${totalTiles}`);
  logger.info(`  ✓ Exported    : ${totalExported}`);
  logger.info(`  ⚠ Skipped     : ${totalSkipped}  (already exists or source missing)`);
  logger.info(`  ✗ Failed      : ${totalFailed}`);
  logger.info(`${"═".repeat(60)}`);

  allResults.forEach((r) => {
    logger.info(`\n  Region: ${r.regionName}`);
    r.exportResults
      .filter((t) => t.state === "COMPLETED")
      .forEach((t) => {
        logger.info(`    ✓ ${t.assetId}`);
        logger.info(`      Task ID : ${t.taskId}`);
      });
    r.errors.forEach((e) => {
      logger.warn(`    ✗ ${e.tileId} — ${e.error}`);
    });
  });

  logger.info(`\n  Load any feature tile in GEE via:`);
  logger.info(`  ee.Image("${OUTPUT_ASSET_ROOT}/<tileName>_features_${DATA_YEAR}")`);
  logger.info(`\n  Band index reference for CNN input tensor:`);
  OUTPUT_BAND_ORDER.forEach((band, i) => {
    logger.info(`    [${i}] ${band}`);
  });
  logger.info(`${"═".repeat(60)}\n`);

  if (allErrors.length === Object.keys(STUDY_REGIONS).length) {
    logger.error("All regions failed. Exiting with error code.");
    process.exit(1);
  }
}

// ─── Exports ──────────────────────────────────────────────────────────────────
export {
  processRegion,
  processTile,
  buildFeatureStack,
  computeNDVI,
  computeNDWI,
  submitExportTask,
  pollExportTask,
  createAssetFolderIfMissing,
  assetExists,
  loadAndValidateAsset,
  generateTiles,
  validateBoundingBox,
  estimateAreaKm2,
  sanitiseAssetName,
  resolveS2AssetId,
  resolveS1AssetId,
  resolveOutputAssetId,
  STUDY_REGIONS,
  OUTPUT_BAND_ORDER,
  S2_BANDS,
  S1_BANDS,
  S2_ASSET_ROOT,
  S1_ASSET_ROOT,
  OUTPUT_ASSET_ROOT,
  DATA_YEAR,
  MAX_TILE_AREA_KM2,
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