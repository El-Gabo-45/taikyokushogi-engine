import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    port: 5000,
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
})