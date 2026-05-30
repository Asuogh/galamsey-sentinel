/**
 * @file nechad-algorithm.js
 * @description Phase 4 — Water Turbidity Monitoring via the Nechad (2010)
 *              Suspended Particulate Matter (SPM) algorithm.
 *
 *              Loads 8-band processed feature composites from Earth Engine
 *              Assets, isolates river pixels using an NDWI-based water mask,
 *              computes SPM concentration in mg/L for each water pixel using
 *              the Nechad algorithm, and reports mean/max/std turbidity
 *              statistics for a defined river segment via reduceRegion().
 *
 * REPORT CONTEXT — What is Water Turbidity and Why Does It Matter?
 * ─────────────────────────────────────────────────────────────────────────────
 * Turbidity is a measure of water clarity — specifically, how much suspended
 * particulate matter (SPM) scatters light as it passes through water.
 * SPM includes fine mineral sediments, organic particles, and algae.
 *
 * In the context of Galamsey (illegal artisanal gold mining), turbidity is
 * the most direct and visible indicator of mining activity on river systems.
 * The mining process involves:
 *   1. Excavating and washing alluvial sediment to extract gold particles.
 *   2. Discharging the sediment-laden wastewater directly into nearby rivers.
 *   3. The fine clay and silt particles remain suspended for days to weeks,
 *      turning the water bright yellow or brown and dramatically increasing
 *      SPM concentrations from background levels of 5–20 mg/L to values
 *      exceeding 500 mg/L in severely affected reaches.
 *
 * Monitoring SPM via satellite provides a synoptic, repeated, and objective
 * measurement of pollution levels across the Pra, Ankobra, and Birim rivers
 * that cannot be achieved cost-effectively by ground-based sampling alone.
 *
 * REPORT CONTEXT — The Nechad (2010) SPM Algorithm:
 * ─────────────────────────────────────────────────────────────────────────────
 * The Nechad algorithm (Nechad et al., 2010, Remote Sensing of Environment)
 * is an empirically-calibrated semi-analytical model that retrieves SPM
 * concentration from a single visible/NIR band's surface reflectance:
 *
 *   SPM = (A_SPM × ρ) / (1 − ρ / C_SPM) + B_SPM
 *
 * Where:
 *   ρ      = normalised water-leaving surface reflectance of the chosen band
 *   A_SPM  = empirical scaling coefficient (band-specific)
 *   C_SPM  = normalised reflectance at saturation (theoretical maximum ~0.1791)
 *   B_SPM  = additive offset (corrects for sensor noise / atmospheric residuals)
 *
 * The formula is derived from the bio-optical theory of backscattering in
 * turbid waters, where SPM concentration is proportional to the ratio of
 * backscattering to absorption. The division by (1 − ρ/C_SPM) accounts for
 * the non-linear saturation of reflectance at very high SPM concentrations.
 *
 * REPORT CONTEXT — Why the Red Band (B4)?
 * ─────────────────────────────────────────────────────────────────────────────
 * The Nechad algorithm can be applied to multiple spectral bands. We use B4
 * (Red, 665nm) for the following reasons:
 *
 *  1. SPECTRAL RESPONSE: Inorganic mineral sediments (the dominant SPM type
 *     in Galamsey-affected rivers) have high reflectance in the red band.
 *     The red/near-red portion of the spectrum is the most sensitive region
 *     for SPM detection in moderately to highly turbid inland waters.
 *
 *  2. CALIBRATION AVAILABILITY: Nechad et al. (2010) published calibrated
 *     A_SPM and C_SPM coefficients specifically for the red band that have
 *     been validated across multiple turbid coastal and inland water bodies.
 *     These published coefficients are directly applicable to Sentinel-2 B4.
 *
 *  3. DYNAMIC RANGE: For SPM > 100 mg/L (severely impacted Galamsey rivers),
 *     the red band remains within a measurable, non-saturated reflectance
 *     range (ρ < 0.15) while shorter wavelengths (Blue, Green) saturate.
 *
 * Published Nechad coefficients for the Red band (Nechad et al. 2010, Table 3):
 *   A_SPM = 355.85  [mg/L per dimensionless reflectance unit]
 *   C_SPM = 1.74    [dimensionless, unitless reflectance scaling]
 *   B_SPM = 0       [mg/L, offset — set to 0 for standard inland water use]
 *
 * REPORT CONTEXT — NDWI Water Mask:
 * ─────────────────────────────────────────────────────────────────────────────
 * We use the NDWI band from our Phase 2.5 feature stack (already computed as
 * (B3 − B8) / (B3 + B8) by feature-engineer.js) to create a binary water mask.
 *
 * Standard clear water threshold: NDWI > 0
 * Turbid/muddy water threshold  : NDWI > −0.1  (recommended for Ghana rivers)
 *
 * Galamsey-impacted rivers carry extremely high sediment loads that suppress
 * the NDWI signal. The NIR band (B8) is elevated by the suspended sediments,
 * pushing NDWI toward zero or slightly negative values even for open water.
 * Using a threshold of −0.1 rather than 0 ensures we capture turbid river
 * pixels that would be missed by the standard threshold, at the cost of
 * slightly more false positives on wet soil at river banks.
 *
 * We combine the NDWI mask with a B4 reflectance plausibility check
 * (0 < ρ_B4 < C_SPM) to exclude pixels where the Nechad formula would
 * produce physically impossible (negative or infinite) SPM values.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * SPM Interpretation Scale for Ghana Rivers (Galamsey context):
 *   <  20 mg/L  → Clean / background turbidity
 *   20– 50 mg/L → Slightly turbid (seasonal runoff, light mining upstream)
 *   50–100 mg/L → Moderately turbid (active mining influence)
 *  100–300 mg/L → Highly turbid (direct Galamsey discharge nearby)
 *   > 300 mg/L  → Extremely turbid (active pit discharge into channel)
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @module nechad-algorithm
 */

import ee             from "@google/earthengine";
import path           from "path";
import { fileURLToPath } from "url";
import "dotenv/config";
import { authenticateGEE, logger } from "../ingestion/gee-auth.js";

// ─── ES Module __dirname Shim ─────────────────────────────────────────────────
const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 1 — NECHAD ALGORITHM COEFFICIENTS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Nechad (2010) SPM retrieval coefficients for the Sentinel-2 Red band (B4).
 *
 * Source: Nechad, B., Ruddick, K.G., Park, Y., 2010.
 *         "Calibration and validation of a generic multisensor algorithm for
 *          mapping of total suspended matter in turbid waters."
 *         Remote Sensing of Environment, 114(4), pp.854–866.
 *         Table 3, Red band (660–680nm), MERIS/Sentinel-2 compatible.
 *
 * To use an alternative band, update these three constants and change
 * SPM_INPUT_BAND to the corresponding band name (e.g. "B3" for Green).
 */

/**
 * A_SPM: Calibration coefficient (mg/L).
 * Physically represents the SPM-to-reflectance conversion factor.
 * Higher values mean more SPM per unit of reflectance — reflects
 * the optical efficiency of the sediment type being modelled.
 */
const NECHAD_A = 355.85;

/**
 * C_SPM: Normalised reflectance at saturation (dimensionless).
 * Represents the theoretical maximum reflectance of an infinitely
 * turbid water body. Pixels with ρ ≥ C_SPM would produce infinite
 * or negative SPM values — these are masked out as physically invalid.
 */
const NECHAD_C = 1.74;

/**
 * B_SPM: Additive offset (mg/L).
 * Corrects for any systematic bias in the atmospheric correction.
 * Set to 0 for standard inland water retrievals as per Nechad et al. (2010).
 * Can be adjusted if a site-specific calibration dataset is available.
 */
const NECHAD_B = 0;

/**
 * The Sentinel-2 band used as input to the Nechad SPM formula.
 * Must be present in the 8-band feature stack (BAND_NAMES).
 */
const SPM_INPUT_BAND = "B4";

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 2 — WATER MASK CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * NDWI threshold for water pixel identification.
 *
 * REPORT CONTEXT — Threshold Selection for Turbid Rivers:
 * ─────────────────────────────────────────────────────────────────────────────
 * We use −0.1 rather than the standard 0.0 threshold because:
 *
 *  • Galamsey-impacted rivers in Ghana have NDWI values that cluster around
 *    −0.05 to +0.10, compared to clean water which typically gives NDWI > 0.2.
 *
 *  • The high suspended sediment load increases NIR reflectance (B8), which
 *    is the denominator term in NDWI = (B3 − B8) / (B3 + B8), pulling the
 *    index toward zero and below.
 *
 *  • A threshold of −0.1 captures the turbid river pixels we need to analyse
 *    while still excluding dry soil (NDWI typically < −0.3) and vegetation
 *    (NDWI typically < −0.2).
 *
 * If you find the water mask is over-extending onto wet riverbanks, raise
 * this value toward 0. If river pixels are being excluded, lower it toward
 * −0.2. The optimal value is site-specific and can be calibrated against
 * field measurements.
 * ─────────────────────────────────────────────────────────────────────────────
 */
const NDWI_WATER_THRESHOLD = -0.1;

/**
 * Minimum valid B4 reflectance for SPM computation.
 * Pixels below this value are near-zero reflectance (deep clear water or
 * shadow) and may produce unreliable SPM estimates.
 */
const B4_MIN_REFLECTANCE = 0.001;

/**
 * Reflectance normalisation divisor for Sentinel-2 Level-2A products.
 *
 * REPORT CONTEXT — Why divide by 10000?
 * ─────────────────────────────────────────────────────────────────────────────
 * Sentinel-2 L2A Surface Reflectance values in GEE are stored as scaled
 * integers in the range [0, 10000], where 10000 represents a reflectance
 * of 1.0 (100%). The Nechad algorithm requires physical reflectance in the
 * range [0, 1]. We divide by 10000 to convert from the digital number
 * scale to physical reflectance units before applying the formula.
 *
 * Note: Our median composites were built from L2A imagery, so this scaling
 * factor is already embedded in the exported asset values. If you ever
 * rebuild the pipeline from L1C (Top-of-Atmosphere) imagery, this divisor
 * would need to change and an atmospheric correction step would be required
 * before turbidity retrieval.
 * ─────────────────────────────────────────────────────────────────────────────
 */
const S2_REFLECTANCE_SCALE = 10000;

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 3 — RIVER SEGMENT DEFINITIONS
//
// HOW TO ADD MORE RIVER SEGMENTS:
// ─────────────────────────────────────────────────────────────────────────────
// Each entry defines a named river segment for turbidity analysis.
//
//   id        : Unique identifier used in log output and future DB storage.
//   name      : Human-readable segment description.
//   centre    : [longitude, latitude] of the segment centre point.
//   bufferKm  : Half-width of the analysis box in kilometres.
//               Larger values capture a longer river reach but may include
//               non-water pixels that dilute the statistics.
//   river     : Name of the parent river for grouping in output.
//   notes     : Context for interpreting results.
//
// To add a new segment, append an entry to the RIVER_SEGMENTS array.
// ═══════════════════════════════════════════════════════════════════════════════

const RIVER_SEGMENTS = [
  {
    id       : "pra_galamsey_hotspot",
    name     : "Pra River — Galamsey Hotspot Reach",
    centre   : [-1.553, 5.598],
    bufferKm : 5,
    river    : "Pra",
    notes    : "Immediately downstream of confirmed active Galamsey pits. " +
               "Expected SPM > 100 mg/L based on field reports.",
  },
  {
    id       : "pra_upstream_reference",
    name     : "Pra River — Upstream Reference Reach",
    centre   : [-1.510, 5.650],
    bufferKm : 5,
    river    : "Pra",
    notes    : "Upstream reference site with minimal direct mining influence. " +
               "Expected SPM 10–30 mg/L. Used as baseline for anomaly detection.",
  },
  // ── Add more river segments below this line ───────────────────────────────
  // {
  //   id       : "ankobra_mid_reach",
  //   name     : "Ankobra River — Mid Reach",
  //   centre   : [-2.1, 5.3],
  //   bufferKm : 5,
  //   river    : "Ankobra",
  //   notes    : "",
  // },
  // {
  //   id       : "birim_oda",
  //   name     : "Birim River — Oda Reach",
  //   centre   : [-0.98, 5.93],
  //   bufferKm : 5,
  //   river    : "Birim",
  //   notes    : "",
  // },
];

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 4 — ASSET & PIPELINE CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/** Source asset root for 8-band processed feature composites (Phase 2.5). */
const FEATURES_ASSET_ROOT = process.env.GEE_FEATURES_ASSET_ROOT ||
  "projects/galamsey-sentinel/assets/processed_features";

/** Data year — must match DATA_YEAR used in feature-engineer.js. */
const DATA_YEAR = parseInt(process.env.DATA_YEAR) || 2025;

/**
 * Scale in metres for reduceRegion() statistics computation.
 *
 * REPORT CONTEXT — Why 30m for statistics, not 10m?
 * ─────────────────────────────────────────────────────────────────────────────
 * We use 30m rather than the native 10m scale for the reduceRegion call
 * because:
 *  1. GEE's reduceRegion() is subject to a pixel count limit per call.
 *     At 10m over a 10km² river segment, there are ~100,000 pixels —
 *     well within the limit, but using 30m (~11,000 pixels) is faster.
 *  2. Mean and max statistics are robust to this modest resampling; the
 *     3× aggregation averages out sub-pixel noise in the reflectance data.
 *  3. For final publication or dashboard display, SPM statistics are
 *     reported as segment-level averages, not pixel-level maps.
 * Set to 10 if you need pixel-exact statistics for academic validation.
 * ─────────────────────────────────────────────────────────────────────────────
 */
const STATS_SCALE_METRES = 30;

/** Maximum pixels for reduceRegion(). Raise if a large segment errors. */
const MAX_PIXELS = 1e8;

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 5 — STUDY REGIONS & TILING (reproduced for asset tile lookup)
// ═══════════════════════════════════════════════════════════════════════════════

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
// SECTION 6 — UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

/** @param {number} ms @returns {Promise<void>} */
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** @param {string} name @returns {string} */
function sanitiseName(name) {
  return name
    .replace(/[^a-zA-Z0-9_-]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "");
}

/**
 * Returns a human-readable turbidity severity label for an SPM value.
 *
 * @param {number|null} spmMgL - SPM concentration in mg/L, or null if invalid.
 * @returns {string} Severity label.
 */
function spmSeverityLabel(spmMgL) {
  if (spmMgL === null || spmMgL === undefined || isNaN(spmMgL)) {
    return "NO DATA";
  }
  if (spmMgL <  20) return "CLEAN           (< 20 mg/L)";
  if (spmMgL <  50) return "SLIGHTLY TURBID (20–50 mg/L)";
  if (spmMgL < 100) return "MODERATELY TURBID (50–100 mg/L)";
  if (spmMgL < 300) return "HIGHLY TURBID   (100–300 mg/L)";
  return               "EXTREMELY TURBID (> 300 mg/L) ⚠ ALERT";
}

/**
 * Formats an SPM value for terminal display.
 * @param {number|null} value
 * @param {number}      [decimals=2]
 * @returns {string}
 */
function formatSPM(value, decimals = 2) {
  if (value === null || value === undefined || isNaN(value)) return "N/A";
  return `${Number(value).toFixed(decimals)} mg/L`;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 7 — FEATURE COMPOSITE LOADER
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Loads the processed feature asset that covers a given [lon, lat] coordinate.
 * Reconstructs the tile grid deterministically (same logic as upstream scripts).
 *
 * @param {number} lon
 * @param {number} lat
 * @param {string} segmentId - For log messages.
 * @returns {Promise<{ image: ee.Image, tileId: string, assetId: string }>}
 */
async function loadFeatureCompositeForPoint(lon, lat, segmentId) {
  const tag = `[LOAD][${segmentId}]`;

  const allTiles = [];
  for (const [, region] of Object.entries(STUDY_REGIONS)) {
    generateTiles(region.bbox, region.name).forEach((t) => allTiles.push(t));
  }

  const EPSILON = 0.0001;
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
      `Check that the river segment centre falls within a study region bbox.`
    );
  }

  const assetName = sanitiseName(`${matchingTile.tileId}_features_${DATA_YEAR}`);
  const assetId   = `${FEATURES_ASSET_ROOT}/${assetName}`;

  logger.info(`${tag} Point [${lon}, ${lat}] → tile: ${matchingTile.tileId}`);
  logger.info(`${tag} Asset: ${assetId}`);

  const exists = await new Promise((resolve) => {
    ee.data.getAsset(assetId, (result, error) => resolve(!error && result != null));
  });

  if (!exists) {
    throw new Error(
      `${tag} Feature asset not found: "${assetId}". ` +
      `Ensure feature-engineer.js completed for tile "${matchingTile.tileId}".`
    );
  }

  const image = ee.Image(assetId);

  // Validate that B4 and NDWI are present — both required for turbidity.
  const bandNames = await new Promise((resolve, reject) => {
    image.bandNames().evaluate((result, error) => {
      if (error) reject(new Error(`${tag} Band validation failed: ${error}`));
      else resolve(result);
    });
  });

  const required = [SPM_INPUT_BAND, "NDWI"];
  const missing  = required.filter((b) => !bandNames.includes(b));
  if (missing.length > 0) {
    throw new Error(
      `${tag} Asset missing required bands: [${missing.join(", ")}]. ` +
      `Re-run feature-engineer.js for tile "${matchingTile.tileId}".`
    );
  }

  logger.info(`${tag} Asset validated. Bands: [${bandNames.join(", ")}]`);

  return { image, tileId: matchingTile.tileId, assetId };
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 8 — WATER MASK
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Creates a binary water mask from the NDWI band of a feature composite.
 *
 * REPORT CONTEXT — Water Mask Logic:
 * ─────────────────────────────────────────────────────────────────────────────
 * The mask combines two conditions to identify valid water pixels:
 *
 *  Condition 1 — NDWI threshold:
 *    NDWI > NDWI_WATER_THRESHOLD (−0.1)
 *    Selects pixels that are spectrally consistent with open water, including
 *    turbid/muddy water surfaces that depress NDWI below 0.
 *
 *  Condition 2 — B4 reflectance plausibility:
 *    ρ_B4 > B4_MIN_REFLECTANCE   (exclude near-zero / shadow pixels)
 *    ρ_B4 < C_SPM (NECHAD_C)     (exclude pixels where Nechad formula diverges)
 *
 * Only pixels satisfying BOTH conditions are considered valid water pixels
 * for the SPM computation. The .and() operator in GEE performs a bitwise AND
 * on the binary mask images.
 *
 * The final .selfMask() call converts 0-valued (non-water) pixels to masked
 * (transparent) pixels so that reduceRegion() automatically ignores them.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {ee.Image} featureImage - 8-band feature composite with NDWI and B4.
 * @returns {{ waterMask: ee.Image, b4Normalised: ee.Image }}
 */
function buildWaterMask(featureImage) {
  // ── Normalise B4 from DN scale [0, 10000] to physical reflectance [0, 1] ──
  // This is required by the Nechad formula which expects ρ in [0, 1].
  const b4Normalised = featureImage
    .select(SPM_INPUT_BAND)
    .divide(S2_REFLECTANCE_SCALE)
    .rename("B4_rho");

  // ── NDWI-based water condition ────────────────────────────────────────────
  const ndwiWater = featureImage
    .select("NDWI")
    .gt(NDWI_WATER_THRESHOLD); // 1 where NDWI > −0.1, else 0

  // ── B4 plausibility conditions ────────────────────────────────────────────
  // Lower bound: exclude shadow / near-zero pixels
  const b4AboveMin = b4Normalised.gt(B4_MIN_REFLECTANCE);

  // Upper bound: exclude pixels where ρ ≥ C_SPM (Nechad formula diverges)
  // We use a conservative ceiling of 0.95 × C_SPM to leave a safety margin.
  const b4BelowSat = b4Normalised.lt(NECHAD_C * 0.95);

  // ── Combine all conditions ─────────────────────────────────────────────────
  const waterMask = ndwiWater
    .and(b4AboveMin)
    .and(b4BelowSat)
    .selfMask(); // mask 0-pixels so reducer ignores them

  return { waterMask, b4Normalised };
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 9 — NECHAD SPM COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Applies the Nechad (2010) SPM algorithm to a normalised reflectance image
 * and masks the result to water pixels only.
 *
 * REPORT CONTEXT — SPM Formula Implementation:
 * ─────────────────────────────────────────────────────────────────────────────
 * SPM (mg/L) = (A × ρ) / (1 − ρ / C) + B
 *
 * Implemented in GEE using server-side ee.Image arithmetic operations:
 *
 *  Step 1: numerator   = A_SPM × ρ           → ee.Image.multiply(NECHAD_A)
 *  Step 2: ratio       = ρ / C_SPM           → ee.Image.divide(NECHAD_C)
 *  Step 3: denominator = 1 − (ρ / C_SPM)    → ee.Image.subtract(ratio) on constant(1)
 *  Step 4: spm_raw     = numerator / denominator + B_SPM
 *  Step 5: Apply water mask → only river pixels retain values
 *  Step 6: Rename output band to "SPM_mgL" for clarity in downstream analysis
 *
 * All arithmetic is performed on server-side ee.Image objects (lazy evaluation)
 * — no pixel data is transferred to our Node.js process until reduceRegion()
 * triggers the computation.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {ee.Image} b4Normalised - Single-band image of B4 reflectance ρ ∈ [0, 1].
 * @param {ee.Image} waterMask    - Binary mask: 1 = valid water pixel.
 * @returns {ee.Image} SPM image in mg/L, masked to water pixels, band "SPM_mgL".
 */
function computeSPM(b4Normalised, waterMask) {
  // Step 1: numerator = A_SPM × ρ
  const numerator   = b4Normalised.multiply(NECHAD_A);

  // Step 2: ratio = ρ / C_SPM
  const ratio       = b4Normalised.divide(NECHAD_C);

  // Step 3: denominator = 1 − (ρ / C_SPM)
  const denominator = ee.Image.constant(1).subtract(ratio);

  // Step 4: spm = numerator / denominator + B_SPM
  const spmImage    = numerator
    .divide(denominator)
    .add(NECHAD_B)
    .rename("SPM_mgL");

  // Step 5: Apply water mask — non-water pixels are set to masked (null)
  const spmMasked = spmImage.updateMask(waterMask);

  return spmMasked;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 10 — RIVER SEGMENT STATISTICS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Builds the river segment analysis region as a GEE geometry.
 * The segment is defined as a square bounding box around the centre point,
 * with sides of length 2 × bufferKm kilometres.
 *
 * @param {object} segment - A RIVER_SEGMENTS entry.
 * @returns {ee.Geometry.Rectangle}
 */
function buildSegmentGeometry(segment) {
  const [lon, lat]  = segment.centre;
  const degLat      = segment.bufferKm / 111;
  const degLon      = segment.bufferKm / (111 * Math.cos((lat * Math.PI) / 180));

  return ee.Geometry.Rectangle([
    lon - degLon,
    lat - degLat,
    lon + degLon,
    lat + degLat,
  ]);
}

/**
 * Computes SPM statistics for a single river segment using reduceRegion().
 *
 * REPORT CONTEXT — reduceRegion() for Turbidity Statistics:
 * ─────────────────────────────────────────────────────────────────────────────
 * ee.Reducer.combine() allows us to compute multiple statistics (mean, max,
 * standard deviation, count) in a SINGLE server-side reduceRegion() call,
 * which is far more efficient than separate calls for each statistic.
 *
 * The reducers are:
 *  • mean()       → Average SPM across all water pixels in the segment
 *                   Represents the "typical" turbidity level of the reach
 *  • max()        → Maximum SPM in any single pixel
 *                   Identifies the most severely impacted location
 *  • stdDev()     → Standard deviation of SPM across water pixels
 *                   High std dev indicates spatially heterogeneous pollution
 *                   (e.g. a point-source plume dispersing into the river)
 *  • count()      → Number of valid water pixels analysed
 *                   Quality control: low count means sparse water coverage
 *
 * bestEffort: true instructs GEE to automatically increase the scale if
 * the pixel count exceeds the internal limit, rather than throwing an error.
 * This ensures large segments always return a result (at potentially coarser
 * resolution) rather than failing silently.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @param {object}    params
 * @param {ee.Image}  params.spmImage  - SPM image in mg/L, water-masked.
 * @param {object}    params.segment   - River segment descriptor.
 * @returns {Promise<{
 *   segmentId   : string,
 *   mean        : number | null,
 *   max         : number | null,
 *   stdDev      : number | null,
 *   pixelCount  : number | null,
 *   severity    : string,
 *   assetId     : string,
 * }>}
 */
async function computeSegmentStats({ spmImage, segment, assetId }) {
  const tag = `[STATS][${segment.id}]`;

  logger.info(`${tag} Computing reduceRegion() statistics...`);
  logger.info(
    `${tag} Segment: ${segment.name} | ` +
    `Centre: [${segment.centre.join(", ")}] | ` +
    `Buffer: ±${segment.bufferKm}km`
  );

  const segmentGeometry = buildSegmentGeometry(segment);

  // Combine mean, max, stdDev, and count into a single reducer call.
  const combinedReducer = ee.Reducer.mean()
    .combine({ reducer2: ee.Reducer.max(),    sharedInputs: true })
    .combine({ reducer2: ee.Reducer.stdDev(), sharedInputs: true })
    .combine({ reducer2: ee.Reducer.count(),  sharedInputs: true });

  // Trigger the server-side computation via .evaluate().
  const statsDict = await new Promise((resolve, reject) => {
    spmImage
      .reduceRegion({
        reducer    : combinedReducer,
        geometry   : segmentGeometry,
        scale      : STATS_SCALE_METRES,
        maxPixels  : MAX_PIXELS,
        bestEffort : true,          // auto-coarsen scale if pixel limit hit
        tileScale  : 4,             // reduce per-tile memory for large regions
      })
      .evaluate((result, error) => {
        if (error) {
          reject(new Error(`${tag} reduceRegion() failed: ${error}`));
        } else {
          resolve(result);
        }
      });
  });

  // GEE's combined reducer appends the stat type to the band name:
  //   "SPM_mgL_mean", "SPM_mgL_max", "SPM_mgL_stdDev", "SPM_mgL_count"
  const mean       = statsDict?.["SPM_mgL_mean"]   ?? null;
  const max        = statsDict?.["SPM_mgL_max"]    ?? null;
  const stdDev     = statsDict?.["SPM_mgL_stdDev"] ?? null;
  const pixelCount = statsDict?.["SPM_mgL_count"]  ?? null;

  const severity = spmSeverityLabel(mean);

  return {
    segmentId  : segment.id,
    segmentName: segment.name,
    river      : segment.river,
    centre     : segment.centre,
    bufferKm   : segment.bufferKm,
    mean,
    max,
    stdDev,
    pixelCount,
    severity,
    assetId,
    notes      : segment.notes,
    dataYear   : DATA_YEAR,
    computedAt : new Date().toISOString(),
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 11 — PER-SEGMENT TURBIDITY PIPELINE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Runs the full turbidity analysis pipeline for a single river segment:
 *   1. Load covering feature composite asset.
 *   2. Build water mask from NDWI and B4.
 *   3. Compute SPM image via Nechad formula.
 *   4. Compute segment statistics via reduceRegion().
 *   5. Return and log the results.
 *
 * @param {object} segment    - A RIVER_SEGMENTS entry.
 * @param {number} [attempt=1]
 * @returns {Promise<object>} Statistics object.
 */
async function analyseSegment(segment, attempt = 1) {
  const tag = `[SEGMENT][${segment.id}]`;
  const [lon, lat] = segment.centre;

  try {
    // ── 1. Load feature composite ──────────────────────────────────────────
    const { image, assetId } = await loadFeatureCompositeForPoint(
      lon, lat, segment.id
    );

    // ── 2. Build water mask ────────────────────────────────────────────────
    logger.info(`${tag} Building water mask (NDWI > ${NDWI_WATER_THRESHOLD})...`);
    const { waterMask, b4Normalised } = buildWaterMask(image);

    // ── 3. Compute SPM via Nechad algorithm ───────────────────────────────
    logger.info(
      `${tag} Applying Nechad SPM formula: ` +
      `(${NECHAD_A} × ρ) / (1 − ρ / ${NECHAD_C}) + ${NECHAD_B}`
    );
    const spmImage = computeSPM(b4Normalised, waterMask);

    // ── 4. Compute segment statistics ─────────────────────────────────────
    const stats = await computeSegmentStats({ spmImage, segment, assetId });

    return stats;

  } catch (err) {
    const isRetryable =
      err.message.toLowerCase().includes("quota")             ||
      err.message.toLowerCase().includes("too many requests") ||
      err.message.toLowerCase().includes("timeout")           ||
      err.message.toLowerCase().includes("network")           ||
      err.message.toLowerCase().includes("rate");

    if (isRetryable && attempt <= 3) {
      const backoffMs = 10_000 * attempt;
      logger.warn(
        `${tag} Transient error (attempt ${attempt}/3): ${err.message}. ` +
        `Retrying in ${backoffMs / 1000}s...`
      );
      await sleep(backoffMs);
      return analyseSegment(segment, attempt + 1);
    }

    throw new Error(`${tag} Failed: ${err.message}`);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 12 — RESULTS LOGGER
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Prints a formatted turbidity report for all analysed segments to the terminal.
 * Groups results by river for readability.
 *
 * @param {object[]} allStats - Array of stats objects from analyseSegment().
 */
function logTurbidityReport(allStats) {
  logger.info(`\n${"═".repeat(70)}`);
  logger.info("GALAMSEY SENTINEL — PHASE 4: TURBIDITY REPORT");
  logger.info(`${"═".repeat(70)}`);
  logger.info(`  Algorithm   : Nechad (2010) SPM — Red Band (B4)`);
  logger.info(`  Coefficients: A=${NECHAD_A}, C=${NECHAD_C}, B=${NECHAD_B}`);
  logger.info(`  Band        : ${SPM_INPUT_BAND} (665nm Red, ${S2_REFLECTANCE_SCALE} DN scale)`);
  logger.info(`  Water mask  : NDWI > ${NDWI_WATER_THRESHOLD} AND 0 < ρ < ${NECHAD_C * 0.95}`);
  logger.info(`  Stats scale : ${STATS_SCALE_METRES}m`);
  logger.info(`  Data year   : ${DATA_YEAR}`);
  logger.info(`  Computed at : ${new Date().toISOString()}`);
  logger.info(`${"─".repeat(70)}`);

  // Group by river name.
  const byRiver = allStats.reduce((acc, s) => {
    const key = s.river ?? "Unknown";
    if (!acc[key]) acc[key] = [];
    acc[key].push(s);
    return acc;
  }, {});

  for (const [river, segments] of Object.entries(byRiver)) {
    logger.info(`\n  ── ${river} River ──────────────────────────────────────────`);

    for (const s of segments) {
      logger.info(`\n  Segment : ${s.segmentName}`);
      logger.info(`  ID      : ${s.segmentId}`);
      logger.info(`  Asset   : ${s.assetId}`);
      logger.info(`  Centre  : [${s.centre.join(", ")}] | Buffer: ±${s.bufferKm}km`);
      logger.info(`  Notes   : ${s.notes}`);
      logger.info(`  ${"─".repeat(50)}`);

      if (s.pixelCount === null || s.pixelCount === 0) {
        logger.warn(`  ⚠ NO WATER PIXELS DETECTED in this segment.`);
        logger.warn(`    Possible causes:`);
        logger.warn(`      • NDWI threshold (${NDWI_WATER_THRESHOLD}) is too strict for this site.`);
        logger.warn(`      • The segment centre does not overlap the river channel.`);
        logger.warn(`      • The feature composite has cloud/mask artefacts over this area.`);
        logger.warn(`    Try: lower NDWI_WATER_THRESHOLD to −0.2, or move the segment centre.`);
        continue;
      }

      logger.info(`  Water pixels analysed : ${s.pixelCount?.toFixed(0) ?? "N/A"}`);
      logger.info(`  ┌────────────────────────────────────────────────────┐`);
      logger.info(`  │  MEAN SPM   : ${formatSPM(s.mean).padEnd(12)} ${s.severity}`);
      logger.info(`  │  MAX SPM    : ${formatSPM(s.max).padEnd(12)}`);
      logger.info(`  │  STD DEV   : ${formatSPM(s.stdDev).padEnd(12)} ${
        s.stdDev > 50
          ? "(high spatial variability — possible point-source plume)"
          : "(spatially uniform)"
      }`);
      logger.info(`  └────────────────────────────────────────────────────┘`);

      // Emit an alert banner for critically turbid segments.
      if (s.mean !== null && s.mean > 300) {
        logger.warn(`  ⚠⚠ CRITICAL TURBIDITY ALERT ⚠⚠`);
        logger.warn(`  Mean SPM ${formatSPM(s.mean)} exceeds 300 mg/L threshold.`);
        logger.warn(`  Likely cause: active Galamsey discharge upstream.`);
        logger.warn(`  Recommended action: dispatch field team / notify Ghana EPA.`);
      } else if (s.mean !== null && s.mean > 100) {
        logger.warn(`  ⚠ HIGH TURBIDITY WARNING`);
        logger.warn(`  Mean SPM ${formatSPM(s.mean)} indicates significant mining influence.`);
      }
    }
  }

  // Cross-segment comparative summary.
  const validStats = allStats.filter((s) => s.mean !== null);
  if (validStats.length > 1) {
    logger.info(`\n${"─".repeat(70)}`);
    logger.info("  CROSS-SEGMENT COMPARISON");
    logger.info(`${"─".repeat(70)}`);

    const sorted = [...validStats].sort((a, b) => (b.mean ?? 0) - (a.mean ?? 0));
    sorted.forEach((s, i) => {
      logger.info(
        `  ${i + 1}. ${s.segmentName.padEnd(45)} ` +
        `Mean: ${formatSPM(s.mean).padEnd(14)} ${spmSeverityLabel(s.mean)}`
      );
    });

    const mostPolluted = sorted[0];
    const cleanest     = sorted[sorted.length - 1];
    const ratio        = mostPolluted.mean / (cleanest.mean || 1);

    logger.info(`\n  Most polluted : ${mostPolluted.segmentName} (${formatSPM(mostPolluted.mean)})`);
    logger.info(`  Cleanest      : ${cleanest.segmentName} (${formatSPM(cleanest.mean)})`);
    logger.info(`  Pollution ratio: ${ratio.toFixed(1)}× difference between sites`);

    if (ratio > 5) {
      logger.warn(
        `\n  ⚠ Large spatial gradient detected (${ratio.toFixed(1)}×). ` +
        `This is consistent with a localised pollution source ` +
        `(mining discharge) between the two measurement sites.`
      );
    }
  }

  logger.info(`\n${"═".repeat(70)}\n`);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 13 — MAIN ORCHESTRATOR
// ═══════════════════════════════════════════════════════════════════════════════

async function main() {
  logger.info("═".repeat(70));
  logger.info("GALAMSEY SENTINEL — PHASE 4: WATER TURBIDITY MONITORING");
  logger.info("═".repeat(70));
  logger.info(`Algorithm   : Nechad (2010) SPM`);
  logger.info(`Input band  : ${SPM_INPUT_BAND} (Red, 665nm)`);
  logger.info(`Coefficients: A_SPM=${NECHAD_A} | C_SPM=${NECHAD_C} | B_SPM=${NECHAD_B}`);
  logger.info(`Water mask  : NDWI > ${NDWI_WATER_THRESHOLD}`);
  logger.info(`Asset root  : ${FEATURES_ASSET_ROOT}`);
  logger.info(`Data year   : ${DATA_YEAR}`);
  logger.info(`Segments    : ${RIVER_SEGMENTS.length}`);
  logger.info("─".repeat(70));

  // ── 1. Authenticate ───────────────────────────────────────────────────────
  try {
    await authenticateGEE();
  } catch (authError) {
    logger.error(`Fatal: GEE authentication failed. ${authError.message}`);
    process.exit(1);
  }

  // ── 2. Analyse each river segment ─────────────────────────────────────────
  const allStats = [];
  const errors   = [];

  for (let i = 0; i < RIVER_SEGMENTS.length; i++) {
    const segment = RIVER_SEGMENTS[i];

    logger.info(
      `\n── Segment ${i + 1}/${RIVER_SEGMENTS.length}: ${segment.name} ──`
    );

    try {
      const stats = await analyseSegment(segment);
      allStats.push(stats);
      logger.info(
        `[${segment.id}] ✓ Complete. ` +
        `Mean SPM: ${formatSPM(stats.mean)} | ` +
        `Max: ${formatSPM(stats.max)} | ` +
        `Pixels: ${stats.pixelCount?.toFixed(0) ?? "N/A"}`
      );
    } catch (segError) {
      logger.error(`[${segment.id}] Failed: ${segError.message}`);
      errors.push({ segmentId: segment.id, error: segError.message });
    }

    // Brief pause between segments to avoid GEE rate limits.
    if (i < RIVER_SEGMENTS.length - 1) {
      await sleep(2000);
    }
  }

  // ── 3. Print the full turbidity report ───────────────────────────────────
  if (allStats.length > 0) {
    logTurbidityReport(allStats);
  }

  // ── 4. Exit summary ───────────────────────────────────────────────────────
  logger.info(`Segments analysed : ${allStats.length} / ${RIVER_SEGMENTS.length}`);
  if (errors.length > 0) {
    logger.warn(`Segments failed   : ${errors.length}`);
    errors.forEach((e) => logger.warn(`  ✗ ${e.segmentId} — ${e.error}`));
  }

  if (errors.length === RIVER_SEGMENTS.length) {
    logger.error("All segments failed. Exiting with error code.");
    process.exit(1);
  }
}

// ─── Exports ──────────────────────────────────────────────────────────────────
export {
  RIVER_SEGMENTS,
  NECHAD_A,
  NECHAD_C,
  NECHAD_B,
  SPM_INPUT_BAND,
  NDWI_WATER_THRESHOLD,
  S2_REFLECTANCE_SCALE,
  FEATURES_ASSET_ROOT,
  DATA_YEAR,
  buildWaterMask,
  computeSPM,
  computeSegmentStats,
  buildSegmentGeometry,
  analyseSegment,
  logTurbidityReport,
  spmSeverityLabel,
  formatSPM,
  loadFeatureCompositeForPoint,
  generateTiles,
  sanitiseName,
};

// Run main() only when executed directly.
const isMain =
  process.argv[1] &&
  fileURLToPath(import.meta.url) === path.resolve(process.argv[1]);

if (isMain) {
  main().catch((err) => {
    logger.error(`Unhandled error in main(): ${err.message}`);
    process.exit(1);
  });
}