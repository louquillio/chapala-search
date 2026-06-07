import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  root: 'src',
  base: './',
  build: {
    outDir: '../docs',
    emptyOutDir: true,
  },
  publicDir: '../public',
});
