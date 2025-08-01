import path from 'path';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');

return {
  define: {
    'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
    'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
  },
  server: {
    host: '0.0.0.0',
    port: 56872,
    cors: true
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '.'),
    }
  }
};


});
