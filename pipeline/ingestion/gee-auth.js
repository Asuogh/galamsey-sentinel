/**
 * @file gee-auth.js
 * @description Google Earth Engine (GEE) Service Account Authentication Module.
 *
 * REPORT CONTEXT — How GEE Authentication Works:
 * ─────────────────────────────────────────────────────────────────────────────
 * Google Earth Engine provides a massive catalogue of satellite datasets,
 * including the complete Sentinel-2 archive, and exposes a computational
 * platform for processing them. To access this platform from a Node.js
 * server (i.e., a non-browser, server-side environment), GEE requires
 * authentication via a Google Cloud "Service Account."
 *
 * A Service Account is a special, non-human Google identity that belongs to
 * an application rather than a person. The authentication flow works as
 * follows:
 *
 *  1. A Service Account is created in Google Cloud Console and registered
 *     with Google Earth Engine at signup.earthengine.google.com.
 *  2. A private key is generated and downloaded as a JSON file containing
 *     a client_email and a private_key (RSA PEM string).
 *  3. At runtime, our Node.js pipeline reads this JSON and passes it
 *     directly to ee.data.authenticateViaPrivateKey(). Internally, the
 *     GEE library uses the key to sign a JWT and exchanges it with
 *     Google's OAuth 2.0 server for a short-lived access token (~1 hour).
 *  4. The library automatically attaches this token to every subsequent
 *     API request — no further manual token management is needed.
 *
 * IMPORTANT — Node.js vs Browser SDK difference:
 * ─────────────────────────────────────────────────────────────────────────────
 * The @google/earthengine package ships two builds:
 *   • Browser build  → exposes ee.ServiceAccountCredentials (a constructor)
 *   • Node.js build  → does NOT expose that constructor; instead uses
 *                      ee.data.authenticateViaPrivateKey(), which accepts
 *                      the raw parsed key object directly.
 *
 * Using the browser constructor in Node.js throws:
 *   "ee.ServiceAccountCredentials is not a constructor"
 *
 * This file uses the correct Node.js method exclusively.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * @module gee-auth
 */

import ee from "@google/earthengine";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import "dotenv/config";
import { createLogger, format, transports } from "winston";

// ─── ES Module __dirname Shim ─────────────────────────────────────────────────
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ─── Logger Setup ─────────────────────────────────────────────────────────────
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

// ─── Private Key Loader ───────────────────────────────────────────────────────
/**
 * Loads and validates the GEE Service Account private key from disk.
 *
 * @returns {{ client_email: string, private_key: string }} Parsed key object.
 * @throws {Error} If the env variable is missing, the file does not exist,
 *                 the JSON is malformed, or required fields are absent.
 */
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

// ─── Main Authentication Function ─────────────────────────────────────────────
/**
 * Authenticates with Google Earth Engine via ee.data.authenticateViaPrivateKey()
 * and then initialises the ee client library.
 *
 * REPORT CONTEXT — Why we wrap callbacks in Promises:
 * ─────────────────────────────────────────────────────────────────────────────
 * The @google/earthengine library predates modern async/await patterns. Its
 * authenticate and initialize methods use Node.js-style error-first callbacks.
 * We wrap each in a Promise so the entire pipeline can use async/await with
 * standard try/catch error handling instead of deeply nested callbacks.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Must be called and awaited once before any other ee.* API call is made.
 *
 * @async
 * @returns {Promise<void>} Resolves when auth and initialisation are complete.
 * @throws {Error} On invalid key, unregistered service account, or network
 *                 failure reaching Google's OAuth servers.
 */
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

  // ── Step 1: Authenticate via Private Key ──────────────────────────────────
  // ee.data.authenticateViaPrivateKey() is the correct Node.js method.
  // It accepts the parsed key object directly — no constructor wrapping needed.
  // Internally it signs a JWT with the private_key field and exchanges it
  // with Google's OAuth 2.0 token endpoint for a bearer access token.
  await new Promise((resolve, reject) => {
    ee.data.authenticateViaPrivateKey(
      privateKey,
      () => {
        // Success callback — the library now holds a valid OAuth access token
        // and will attach it automatically to all subsequent API requests.
        logger.info("GEE OAuth token acquired successfully.");
        resolve();
      },
      (authError) => {
        // Common failure causes:
        //   • Service account not registered at signup.earthengine.google.com
        //   • Private key revoked in Google Cloud Console
        //   • No internet access to reach accounts.google.com
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

  // ── Step 2: Initialise the EE Client ─────────────────────────────────────
  // ee.initialize() must be called after authenticate() and before any
  // ee.Image(), ee.Geometry(), or other ee.* object is constructed.
  // Skipping this step causes cryptic silent failures on all GEE calls.
  await new Promise((resolve, reject) => {
    ee.initialize(
      null, // Use default GEE API endpoint
      null, // Use default asset root
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