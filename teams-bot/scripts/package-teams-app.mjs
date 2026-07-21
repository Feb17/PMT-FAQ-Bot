import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const root = path.resolve(process.cwd());
const appDir = path.join(root, 'appPackage');
const outDir = path.join(root, 'dist', 'teams-app');
const manifestTemplate = fs.readFileSync(path.join(appDir, 'manifest.template.json'), 'utf8');

const env = (name, fallback = '') => (process.env[name] || fallback).trim();
const microsoftAppId = env('MICROSOFT_APP_ID') || env('MicrosoftAppId');
const required = [
  ['TEAMS_APP_ID', env('TEAMS_APP_ID')],
  ['MICROSOFT_APP_ID or MicrosoftAppId', microsoftAppId],
  ['TEAMS_BOT_DOMAIN', env('TEAMS_BOT_DOMAIN')],
];
const missing = required.filter(([, value]) => !value).map(([name]) => name);
if (missing.length) throw new Error(`Missing required env vars: ${missing.join(', ')}`);

const values = {
  TEAMS_APP_ID: env('TEAMS_APP_ID'),
  MICROSOFT_APP_ID: microsoftAppId,
  TEAMS_BOT_DOMAIN: env('TEAMS_BOT_DOMAIN'),
  TEAMS_BOT_DEVELOPER_NAME: env('TEAMS_BOT_DEVELOPER_NAME', 'PMT FAQ Bot'),
  TEAMS_BOT_DEVELOPER_WEBSITE: env('TEAMS_BOT_DEVELOPER_WEBSITE', 'https://example.com'),
  TEAMS_BOT_DEVELOPER_PRIVACY_URL: env('TEAMS_BOT_DEVELOPER_PRIVACY_URL', 'https://example.com/privacy'),
  TEAMS_BOT_DEVELOPER_TERMS_URL: env('TEAMS_BOT_DEVELOPER_TERMS_URL', 'https://example.com/terms'),
  MANIFEST_VERSION: env('MANIFEST_VERSION', '1.0.0'),
  RAG_ASSET_BASE_URL_HOST: new URL(env('RAG_ASSET_BASE_URL', `https://${env('TEAMS_BOT_DOMAIN')}`)).host,
};

fs.rmSync(outDir, { recursive: true, force: true });
fs.mkdirSync(outDir, { recursive: true });
const manifest = manifestTemplate.replace(/\{\{([A-Z0-9_]+)\}\}/g, (_, k) => values[k] || '');
fs.writeFileSync(path.join(outDir, 'manifest.json'), manifest);
fs.copyFileSync(path.join(appDir, 'color.png'), path.join(outDir, 'color.png'));
fs.copyFileSync(path.join(appDir, 'outline.png'), path.join(outDir, 'outline.png'));

try {
  execFileSync('zip', ['-r', path.join(root, 'dist', 'teams-app.zip'), '.'], { cwd: outDir, stdio: 'inherit' });
  console.log(`Packaged ${path.join(root, 'dist', 'teams-app.zip')}`);
} catch {
  console.error(`Rendered ${outDir}. Install 'zip' to create dist/teams-app.zip.`);
}
