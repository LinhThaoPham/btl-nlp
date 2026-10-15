import {defineConfig} from '@playwright/test';
export default defineConfig({
 testDir:'./tests/browser',timeout:60000,expect:{timeout:15000},fullyParallel:false,workers:1,
 use:{baseURL:'http://127.0.0.1:5173',headless:true,launchOptions:{executablePath:process.env.CHROME_PATH||'/opt/google/chrome/chrome',args:['--no-sandbox']}},
 webServer:{command:'npm run dev -- --port 5173 --strictPort',url:'http://127.0.0.1:5173',reuseExistingServer:!process.env.CI},
 reporter:'list'
});
