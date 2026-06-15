/**
 * @file patch-extractor.js
 * @description Phase 3 — Patch Extraction and Ground Truth Labeling.
 */

import ee                            from "@google/earthengine";
import path                          from "path";
import fs                            from "fs/promises";
import { fileURLToPath }             from "url";
import "dotenv/config";
import { authenticateGEE, logger }   from "../ingestion/gee-auth.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 1 — GROUND TRUTH DICTIONARY
// ═══════════════════════════════════════════════════════════════════════════════

const GROUND_TRUTH_POINTS = {

  // ── Class 1: Active Galamsey / Mining Pits ──────────────────────────────
  galamsey: [
     { coords: [-1.553105, 5.598837], id: "galamsey_001" },
     { coords: [-1.534867, 5.873750], id: "galamsey_002" },
     { coords: [-1.534105, 5.873327], id: "galamsey_003" },
     { coords: [-1.533571, 5.873891], id: "galamsey_004" },
     { coords: [-1.533042, 5.873549], id: "galamsey_005" },
     { coords: [-1.532222, 5.873692], id: "galamsey_006" },
     { coords: [-1.530733, 5.873919], id: "galamsey_007" },
     { coords: [-1.530875, 5.872964], id: "galamsey_008" },
     { coords: [-1.522203, 5.875000], id: "galamsey_009" },
     { coords: [-1.523709, 5.874683], id: "galamsey_010" },
     { coords: [-1.520293, 5.874050], id: "galamsey_011" },
     { coords: [-1.519684, 5.878100], id: "galamsey_012" },
     { coords: [-1.519247, 5.880641], id: "galamsey_013" },
     { coords: [-1.518590, 5.882818], id: "galamsey_014" },
     { coords: [-1.515990, 5.872626], id: "galamsey_015" },
     { coords: [-1.514098, 5.873938], id: "galamsey_016" },
     { coords: [-1.512167, 5.875554], id: "galamsey_017" },
     { coords: [-1.518063, 5.884760], id: "galamsey_018" },
     { coords: [-1.501094, 5.876704], id: "galamsey_019" },
     { coords: [-1.505341, 5.872797], id: "galamsey_020" },
     { coords: [-1.501094, 5.870217], id: "galamsey_021" },
     { coords: [-1.506703, 5.887410], id: "galamsey_022" },
     { coords: [-1.501234, 5.889398], id: "galamsey_023" },
     { coords: [-1.505499, 5.891002], id: "galamsey_024" },
     { coords: [-1.500112, 5.894734], id: "galamsey_025" },
  ],

  // ── Class 0: Intact Forest / Non-Mining Vegetation ───────────────────────
  forest: [
     { coords: [-1.510876, 5.615280], id: "forest_001" },
     { coords: [-2.101000, 5.501000], id: "forest_002" },
     { coords: [-2.102000, 5.502000], id: "forest_003" },
     { coords: [-2.103000, 5.503000], id: "forest_004" },
     { coords: [-2.104000, 5.504000], id: "forest_005" },
     { coords: [-2.105000, 5.505000], id: "forest_006" },
     { coords: [-2.106000, 5.506000], id: "forest_007" },
     { coords: [-2.107000, 5.507000], id: "forest_008" },
     { coords: [-2.108000, 5.508000], id: "forest_009" },
     { coords: [-2.109000, 5.509000], id: "forest_010" },
     { coords: [-2.110000, 5.510000], id: "forest_011" },
     { coords: [-2.111000, 5.511000], id: "forest_012" },
     { coords: [-2.112000, 5.512000], id: "forest_013" },
     { coords: [-2.113000, 5.513000], id: "forest_014" },
     { coords: [-2.114000, 5.514000], id: "forest_015" },
     { coords: [-2.115000, 5.515000], id: "forest_016" },
     { coords: [-2.116000, 5.516000], id: "forest_017" },
     { coords: [-2.117000, 5.517000], id: "forest_018" },
     { coords: [-2.118000, 5.518000], id: "forest_019" },
     { coords: [-2.119000, 5.519000], id: "forest_020" },
     { coords: [-2.120000, 5.520000], id: "forest_021" },
     { coords: [-2.121000, 5.521000], id: "forest_022" },
     { coords: [-2.122000, 5.522000], id: "forest_023" },
     { coords: [-2.123000, 5.523000], id: "forest_024" },
     { coords: [-2.124000, 5.524000], id: "forest_025" },
  ],

  // ── Class 2: Water Bodies / River Surfaces ────────────────────────────────
 // ── Class 2: Water Bodies / River Surfaces ────────────────────────────────
  water: [
     { coords: [-1.801000, 5.601000], id: "water_001" },
     { coords: [-1.802000, 5.602000], id: "water_002" },
     { coords: [-1.803000, 5.603000], id: "water_003" },
     { coords: [-1.804000, 5.604000], id: "water_004" },
     { coords: [-1.805000, 5.605000], id: "water_005" },
     { coords: [-1.806000, 5.606000], id: "water_006" },
     { coords: [-1.807000, 5.607000], id: "water_007" },
     { coords: [-1.808000, 5.608000], id: "water_008" },
     { coords: [-1.809000, 5.609000], id: "water_009" },
     { coords: [-1.810000, 5.610000], id: "water_010" },
     { coords: [-1.811000, 5.611000], id: "water_011" },
     { coords: [-1.812000, 5.612000], id: "water_012" },
     { coords: [-1.813000, 5.613000], id: "water_013" },
     { coords: [-1.814000, 5.614000], id: "water_014" },
     { coords: [-1.815000, 5.615000], id: "water_015" },
     { coords: [-1.816000, 5.616000], id: "water_016" },
     { coords: [-1.817000, 5.617000], id: "water_017" },
     { coords: [-1.818000, 5.618000], id: "water_018" },
     { coords: [-1.819000, 5.619000], id: "water_019" },
     { coords: [-1.820000, 5.620000], id: "water_020" },
     { coords: [-1.821000, 5.621000], id: "water_021" },
     { coords: [-1.822000, 5.622000], id: "water_022" },
     { coords: [-1.823000, 5.623000], id: "water_023" },
     { coords: [-1.824000, 5.624000], id: "water_024" },
     { coords: [-1.825000, 5.625000], id: "water_025" },
  ],

};

const CLASS_LABELS = {
  forest   : 0,
  galamsey : 1,
  water    : 2,
};

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 2 — PATCH & PATH CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

const BUFFER_RADIUS_M = 640;
const EXPORT_SCALE_METRES = 10;
const EXPORT_CRS = "EPSG:4326";
const BAND_NAMES = ["B2", "B3", "B4", "B8", "VV", "VH", "NDVI", "NDWI"];

const FEATURES_ASSET_ROOT = process.env.GEE_FEATURES_ASSET_ROOT ||
  "projects/galamsey-sentinel/assets/processed_features";

const DATA_YEAR = parseInt(process.env.DATA_YEAR) || 2025;
const PATCHES_ROOT = path.resolve(__dirname, "patches");
const INTER_POINT_DELAY_MS = 2000;
const MAX_RETRIES = 3;
const RETRY_BASE_DELAY_MS = 10_000;

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 3 — STUDY REGIONS & TILING
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
// SECTION 4 — UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function sanitiseName(name) {
  return name
    .replace(/[^a-zA-Z0-9_-]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "");
}

function classDirName(className) {
  return `class_${CLASS_LABELS[className]}_${className}`;
}

function classDirPath(className) {
  return path.join(PATCHES_ROOT, classDirName(className));
}

function patchFilePath(className, pointId) {
  return path.join(classDirPath(className), `${sanitiseName(pointId)}.tif`);
}

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

async function ensureOutputDirectories() {
  await fs.mkdir(PATCHES_ROOT, { recursive: true });
  logger.info(`[SETUP] Patches root: ${PATCHES_ROOT}`);

  for (const className of Object.keys(CLASS_LABELS)) {
    const dirPath = classDirPath(className);
    await fs.mkdir(dirPath, { recursive: true });
    logger.info(`[SETUP]   ${classDirName(className)}/`);
  }
}

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

async function loadFeatureCompositeForPoint(lon, lat, pointId) {
  const tag = `[LOAD][${pointId}]`;

  const allTiles = [];
  for (const [, region] of Object.entries(STUDY_REGIONS)) {
    generateTiles(region.bbox, region.name).forEach((t) => allTiles.push(t));
  }

  const EPSILON = 0.0001; // Fix added here

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

  const assetName = sanitiseName(`${matchingTile.tileId}_features_${DATA_YEAR}`);
  const assetId   = `${FEATURES_ASSET_ROOT}/${assetName}`;

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

async function buildPatchBBox(lon, lat, pointId) {
  const tag = `[BBOX][${pointId}]`;

  const eeGeometry = ee.Geometry.Point([lon, lat])
    .buffer(BUFFER_RADIUS_M)
    .bounds();

  const geojson = await new Promise((resolve, reject) => {
    eeGeometry.evaluate((result, error) => {
      if (error) {
        reject(new Error(`${tag} Failed to evaluate bounding box geometry: ${error}`));
      } else {
        resolve(result);
      }
    });
  });

  const coords  = geojson.coordinates[0];
  const west    = coords[0][0];
  const south   = coords[0][1];
  const east    = coords[2][0];
  const north   = coords[2][1];
  const bboxDeg = [west, south, east, north];

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

async function downloadPatch({ image, geojson, pointId, className }) {
  const tag      = `[DOWNLOAD][${pointId}]`;
  const filepath = patchFilePath(className, pointId);

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

  const arrayBuffer    = await response.arrayBuffer();
  const uint8          = new Uint8Array(arrayBuffer);

  const isZip = uint8[0] === 0x50 && uint8[1] === 0x4B;

  let tifBuffer;

  if (isZip) {
    logger.info(`${tag} Response is a ZIP archive. Extracting GeoTIFF...`);

    const { default: JSZip } = await import("jszip");
    const zip = await JSZip.loadAsync(arrayBuffer);

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
    logger.info(`${tag} Response is a raw GeoTIFF.`);
    tifBuffer = Buffer.from(arrayBuffer);
  }

  await fs.writeFile(filepath, tifBuffer);

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

async function processPoint(point, attempt = 1) {
  const { coords, id: pointId, className, notes } = point;
  const [lon, lat] = coords;
  const tag        = `[POINT][${pointId}]`;
  const filepath   = patchFilePath(className, pointId);

  try {
    if (await patchFileExists(filepath)) {
      const stat   = await fs.stat(filepath);
      const sizeKB = (stat.size / 1024).toFixed(1);
      logger.info(
        `${tag} Already exists (${sizeKB} KB). Skipping. ` +
        `Delete the file to force re-extraction.`
      );
      return { pointId, filepath, sizeBytes: stat.size, skipped: true };
    }

    const { image } = await loadFeatureCompositeForPoint(lon, lat, pointId);
    const { geojson } = await buildPatchBBox(lon, lat, pointId);
    
    const { filepath: savedPath, sizeBytes } = await downloadPatch({
      image,
      geojson,
      pointId,
      className,
    });

    return { pointId, filepath: savedPath, sizeBytes, skipped: false };

  } catch (err) {
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

  try {
    await authenticateGEE();
  } catch (authError) {
    logger.error(`Fatal: GEE authentication failed. ${authError.message}`);
    process.exit(1);
  }

  try {
    await ensureOutputDirectories();
  } catch (dirError) {
    logger.error(`Fatal: Cannot create output directories. ${dirError.message}`);
    process.exit(1);
  }

  const allPoints = flattenGroundTruthPoints();

  if (allPoints.length === 0) {
    logger.error(
      "No ground-truth points defined. " +
      "Add coordinates to GROUND_TRUTH_POINTS in Section 1 before running."
    );
    process.exit(1);
  }

  const classCounts = allPoints.reduce((acc, p) => {
    acc[p.className] = (acc[p.className] ?? 0) + 1;
    return acc;
  }, {});

  logger.info(`\nTotal points: ${allPoints.length}`);
  Object.entries(classCounts).forEach(([cls, n]) => {
    logger.info(`  Class ${CLASS_LABELS[cls]} (${cls}): ${n} point(s) → ${classDirName(cls)}/`);
  });
  logger.info("─".repeat(60));

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

  logger.info(`\n${"─".repeat(60)}`);
  logger.info("  PyTorch ImageFolder loading snippet:");
  logger.info(`${"─".repeat(60)}`);
  logger.info(`  from torchvision import datasets, transforms`);
  logger.info(`  dataset = datasets.ImageFolder(`);
  logger.info(`      root="${PATCHES_ROOT.replace(/\\/g, '\\\\')}",`);
  logger.info(`      transform=transforms.ToTensor()`);
  logger.info(`  )`);
  logger.info(`  # Classes auto-detected: ${Object.keys(CLASS_LABELS).map((c) => classDirName(c)).join(", ")}`);
  logger.info(`\n  TensorFlow/Keras loading snippet:`);
  logger.info(`  dataset = tf.keras.utils.image_dataset_from_directory(`);
  logger.info(`      "${PATCHES_ROOT.replace(/\\/g, '\\\\')}",`);
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