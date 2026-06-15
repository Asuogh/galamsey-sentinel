
import ee from "@google/earthengine";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import "dotenv/config";
import { createLogger, format, transports } from "winston";


const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


const logger = createLogger({
  level: "info",
  format: format.combine(
    format.timestamp({ format: "YYYY-MM-DD HH:mm:ss" }),
    format.colorize(),
    format.printf(({ timestamp, level, message }) => {
      return `[${timestamp}] [GEE-AUTH] ${level}: ${message}`;
    })
  ),
  transports: [new transports.Console()],
});


function loadPrivateKey() {
  const keyPath = process.env.GEE_SERVICE_ACCOUNT_KEY_PATH;

  if (!keyPath) {
    throw new Error(
      "Environment variable GEE_SERVICE_ACCOUNT_KEY_PATH is not set. " +
        "Add it to your .env file pointing to your GEE service account JSON."
    );
  }

  const absoluteKeyPath = path.resolve(__dirname, "../../", keyPath);

  if (!fs.existsSync(absoluteKeyPath)) {
    throw new Error(
      `GEE private key file not found at: ${absoluteKeyPath}. ` +
        "Ensure the file exists and the path in .env is correct."
    );
  }

  let privateKey;
  try {
    const rawFile = fs.readFileSync(absoluteKeyPath, "utf-8");
    privateKey = JSON.parse(rawFile);
  } catch (parseError) {
    throw new Error(
      `Failed to parse GEE private key JSON: ${parseError.message}. ` +
        "Ensure the file is a valid JSON key downloaded from Google Cloud Console."
    );
  }

  if (!privateKey.client_email || !privateKey.private_key) {
    throw new Error(
      "GEE key JSON is missing 'client_email' and/or 'private_key'. " +
        "Download a fresh key from Google Cloud Console → IAM → Service Accounts."
    );
  }

  return privateKey;
}


async function authenticateGEE() {
  logger.info("Starting GEE Service Account authentication...");

  let privateKey;
  try {
    privateKey = loadPrivateKey();
    logger.info(
      `Private key loaded for service account: ${privateKey.client_email}`
    );
  } catch (keyError) {
    logger.error(`Key loading failed: ${keyError.message}`);
    throw keyError;
  }

  await new Promise((resolve, reject) => {
    ee.data.authenticateViaPrivateKey(
      privateKey,
      () => {
       
        logger.info("GEE OAuth token acquired successfully.");
        resolve();
      },
      (authError) => {
       
        reject(
          new Error(
            `GEE authentication failed: ${authError}. ` +
              "Verify the service account is registered at " +
              "signup.earthengine.google.com and the key is active."
          )
        );
      }
    );
  });

  
  await new Promise((resolve, reject) => {
    ee.initialize(
      null, 
      null,
      () => {
        logger.info(
          "GEE client initialised. Ready to process satellite imagery."
        );
        resolve();
      },
      (initError) => {
        reject(
          new Error(
            `GEE initialisation failed after successful auth: ${initError}. ` +
              "Check https://status.earthengine.google.com for service outages."
          )
        );
      }
    );
  });
}

export { authenticateGEE, logger };

if (process.argv[1].endsWith('gee-auth.js')) {
    
    process.env.GEE_SERVICE_ACCOUNT_KEY_PATH = 'pipeline/ingestion/gee-key.json';
    
    console.log("🔑 Turning the ignition key...");
    authenticateGEE()
        .then(() => console.log("🏁 TEST COMPLETE: You are ready for Phase 2!"))
        .catch(err => console.error("❌ TEST FAILED:", err));
}