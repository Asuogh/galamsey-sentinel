import ee from "@google/earthengine";
import path from "path";
import { fileURLToPath } from "url";
import "dotenv/config";
import { authenticateGEE, logger } from "./gee-auth.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const S1_COLLECTION_ID = "COPERNICUS/S1_GRD";
const TARGET_BANDS = ["VV", "VH"];
const INSTRUMENT_MODE = "IW";
const ORBIT_DIRECTION = "ASCENDING";
const FETCH_YEAR = 2025;

const MAX_TILE_AREA_KM2 = 2000;
const INTER_TILE_DELAY_MS = 3000;
const MAX_RETRIES = 3;
const RETRY_BASE_DELAY_MS = 10_000;

const ASSET_ROOT = process.env.GEE_S1_ASSET_ROOT || "projects/galamsey-sentinel/assets/sentinel1_tiles";
const EXPORT_SCALE_METRES = 10;
const EXPORT_CRS = "EPSG:4326";
const POLL_INTERVAL_MS = 15_000;
const EXPORT_TIMEOUT_MS = parseInt(process.env.GEE_EXPORT_TIMEOUT_MS) || 7_200_000;

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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function estimateAreaKm2(bbox) {
  const [west, south, east, north] = bbox;
  const midLat = (south + north) / 2;
  const latKm  = (north - south) * 111;
  const lonKm  = (east  - west)  * 111 * Math.cos((midLat * Math.PI) / 180);
  return Math.round(latKm * lonKm);
}

function sanitiseAssetName(name) {
  return name
    .replace(/[^a-zA-Z0-9_-]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_|_$/g, "");
}

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

  logger.info(`[${regionName}] Area ~${totalArea} km² exceeds limit. Auto-tiling into ${rows}×${cols} grid (~${estTileArea} km² each).`);

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

async function createAssetFolderIfMissing(assetPath) {
  logger.info(`[ASSET-SETUP] Verifying asset folder: ${assetPath}`);

  await new Promise((resolve, reject) => {
    ee.data.createFolder(assetPath, false, (result, error) => {
      if (!error) {
        logger.info(`[ASSET-SETUP] ✓ Asset folder created: ${assetPath}`);
        return resolve();
      }

      const errorStr = String(error).toLowerCase();
      if (errorStr.includes("already exists") || errorStr.includes("cannot overwrite")) {
        logger.info(`[ASSET-SETUP] ✓ Asset folder already exists: ${assetPath}`);
        return resolve();
      }

      reject(new Error(`[ASSET-SETUP] Failed to create asset folder "${assetPath}": ${error}. Ensure the Cloud Project ID matches your registered GEE project and the service account has the "Earth Engine Resource Writer" IAM role.`));
    });
  });
}

async function assetExists(assetId) {
  return new Promise((resolve) => {
    ee.data.getAsset(assetId, (result, error) => {
      resolve(!error && result != null);
    });
  });
}

async function fetchTile({ bbox, startDate, endDate, tileId, attempt = 1 }) {
  const tag = `[FETCH][${tileId}]`;

  try {
    const [west, south, east, north] = bbox;
    const region = ee.Geometry.Rectangle([west, south, east, north], null, false);

    const collection = ee
      .ImageCollection(S1_COLLECTION_ID)
      .filterBounds(region)
      .filterDate(startDate, endDate)
      .filter(ee.Filter.eq("instrumentMode", INSTRUMENT_MODE))
      .filter(ee.Filter.eq("orbitProperties_pass", ORBIT_DIRECTION))
      .select(TARGET_BANDS);

    const sceneCount = await new Promise((resolve, reject) => {
      collection.size().evaluate((result, error) => {
        if (error) reject(new Error(`GEE evaluate() failed: ${error}`));
        else resolve(result);
      });
    });

    if (sceneCount === 0) {
      logger.warn(`${tag} 0 scenes found. Possible causes: (1) No DESCENDING IW pass covers this tile extent in ${startDate}–${endDate}. (2) Try switching ORBIT_DIRECTION to "ASCENDING" for this region. (3) Verify the tile bbox intersects a Sentinel-1 acquisition track.`);
      return null;
    }

    logger.info(`${tag} ${sceneCount} scenes found (mode: ${INSTRUMENT_MODE}, orbit: ${ORBIT_DIRECTION}). Building median composite...`);

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

    logger.info(`${tag} Composite ready. Bands: [${bandNames.join(", ")}] | Scenes: ${sceneCount} | Mode: ${INSTRUMENT_MODE} | Orbit: ${ORBIT_DIRECTION}`);

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
      logger.warn(`${tag} Transient error (attempt ${attempt}/${MAX_RETRIES}): ${err.message}. Retrying in ${backoffMs / 1000}s...`);
      await sleep(backoffMs);
      return fetchTile({ bbox, startDate, endDate, tileId, attempt: attempt + 1 });
    }

    throw new Error(`${tag} Failed after ${attempt - 1} retries: ${err.message}`);
  }
}

async function submitExportTask({ image, region, metadata, assetRoot = ASSET_ROOT }) {
  const tag = `[EXPORT][${metadata.tileId}]`;

  const year      = new Date(metadata.startDate).getFullYear();
  const assetName = sanitiseAssetName(`${metadata.tileId}_S1_${year}`);
  const assetId   = `${assetRoot}/${assetName}`;

  logger.info(`${tag} Submitting toAsset export.`);
  logger.info(`${tag} Asset destination: ${assetId}`);
  logger.info(`${tag} Scale: ${EXPORT_SCALE_METRES}m | CRS: ${EXPORT_CRS} | Bands: [${metadata.bands.join(", ")}]`);

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
      reject(new Error(`${tag} task.start() failed: ${startError.message}. Common causes: (1) Asset folder "${assetRoot}" does not exist. (2) Asset "${assetId}" already exists — delete it or use a new name. (3) GEE auth token expired — re-run authenticateGEE(). (4) Cloud Project ID in ASSET_ROOT does not match your GEE project.`));
    }
  });

  const taskId = task.id ?? "unknown";

  logger.info(`${tag} Task submitted. GEE Task ID: ${taskId}. Monitor: https://code.earthengine.google.com/tasks`);

  return { taskId, assetId, status: "SUBMITTED" };
}

async function pollExportTask(taskId, assetId) {
  const tag = `[POLL][${assetId.split("/").pop()}]`;
  const startTime = Date.now();

  logger.info(`${tag} Polling every ${POLL_INTERVAL_MS / 1000}s. Timeout: ${EXPORT_TIMEOUT_MS / 60_000} min.`);

  const TERMINAL_STATES = new Set(["COMPLETED", "FAILED", "CANCELLED"]);

  while (true) {
    const elapsedMs = Date.now() - startTime;

    if (elapsedMs > EXPORT_TIMEOUT_MS) {
      logger.warn(`${tag} Timed out after ${Math.round(elapsedMs / 60_000)} min. Check: https://code.earthengine.google.com/tasks`);
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
        const errorMessage = taskStatus?.error_message ?? taskStatus?.description ?? "No details.";
        logger.error(`${tag} ✗ ${state}: ${errorMessage}`);
        return { taskId, state, assetId, elapsedMs, errorMessage };
      }
      return { taskId, state, assetId, elapsedMs };
    }

    await sleep(POLL_INTERVAL_MS);
  }
}

async function fetchSentinel1Composite({ bbox, startDate, endDate, regionName = "unnamed-region" }) {
  logger.info(`\n${"─".repeat(60)}`);
  logger.info(`[${regionName}] Starting Sentinel-1 region fetch + asset export.`);
  logger.info(`[${regionName}] Date: ${startDate} → ${endDate} | Mode: ${INSTRUMENT_MODE} | Orbit: ${ORBIT_DIRECTION} | Bands: ${TARGET_BANDS.join(", ")}`);
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

    logger.info(`\n[${regionName}] ── Tile ${i + 1}/${tiles.length} ──────────────────────`);
    logger.info(`[${regionName}] ID: ${tileId} (row ${row}, col ${col}) | bbox: [${tileBbox.map((v) => v.toFixed(4)).join(", ")}]`);

    try {
      const year            = new Date(startDate).getFullYear();
      const assetName       = sanitiseAssetName(`${tileId}_S1_${year}`);
      const expectedAssetId = `${ASSET_ROOT}/${assetName}`;

      const alreadyExported = await assetExists(expectedAssetId);
      if (alreadyExported) {
        logger.info(`[${tileId}] Asset already exists at ${expectedAssetId}. Skipping.`);
        skippedCount++;
        if (i < tiles.length - 1) await sleep(INTER_TILE_DELAY_MS);
        continue;
      }

      const fetchResult = await fetchTile({ bbox: tileBbox, startDate, endDate, tileId });

      if (fetchResult === null) {
        skippedCount++;
        logger.warn(`[${tileId}] Skipped — no scenes found.`);
        if (i < tiles.length - 1) await sleep(INTER_TILE_DELAY_MS);
        continue;
      }

      const { image, region, metadata } = fetchResult;
      const submission = await submitExportTask({ image, region, metadata });
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
  logger.info(`[${regionName}] ✓ Exported: ${completedCount} | ⚠ Skipped: ${skippedCount} | ✗ Failed: ${errors.length}`);

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

async function main() {
  try {
    await authenticateGEE();
  } catch (authError) {
    logger.error(`Fatal: GEE authentication failed. ${authError.message}`);
    process.exit(1);
  }

  try {
    await createAssetFolderIfMissing(ASSET_ROOT);
  } catch (folderError) {
    logger.error(`Fatal: Asset folder setup failed. ${folderError.message}`);
    process.exit(1);
  }

  const startDate = process.env.FETCH_START_DATE || `${FETCH_YEAR}-01-01`;
  const endDate   = process.env.FETCH_END_DATE   || `${FETCH_YEAR}-12-31`;

  logger.info(`\nSensor      : Sentinel-1 GRD`);
  logger.info(`Mode        : ${INSTRUMENT_MODE} | Orbit: ${ORBIT_DIRECTION}`);
  logger.info(`Bands       : ${TARGET_BANDS.join(", ")}`);
  logger.info(`Date range  : ${startDate} → ${endDate}`);
  logger.info(`Asset root  : ${ASSET_ROOT}`);
  logger.info(`Export scale: ${EXPORT_SCALE_METRES}m | CRS: ${EXPORT_CRS}`);
  logger.info(`Regions     : ${Object.keys(STUDY_REGIONS).join(", ")}\n`);

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
      logger.error(`Fatal error for region "${region.name}": ${regionError.message}`);
      allErrors.push({ regionKey: key, error: regionError.message });
    }
  }

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

  logger.info(`\n  Load any S1 asset in GEE via: ee.Image("${ASSET_ROOT}/<tileName>")`);
  logger.info(`\n  To stack S1 + S2 bands for CNN input:\n  const s2 = ee.Image("projects/galamsey-sentinel/assets/sentinel2_tiles/<tileName>_2025");\n  const s1 = ee.Image("${ASSET_ROOT}/<tileName>_S1_2025");\n  const stacked = s2.addBands(s1); // → [B2, B3, B4, B8, VV, VH]`);
  logger.info(`${"═".repeat(60)}\n`);

  if (allErrors.length === Object.keys(STUDY_REGIONS).length) {
    logger.error("All regions failed. Exiting with error code.");
    process.exit(1);
  }
}

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

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1]);

if (isMain) {
  main().catch((err) => {
    logger.error(`Unhandled error in main(): ${err.message}`);
    process.exit(1);
  });
}