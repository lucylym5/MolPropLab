/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#ffffff",
        foreground: "#1f2937",
        card: "#f8f9fa",
        muted: "#6b7280",
        primary: "#3b82f6",
        accent: "#10b981",
        border: "#e5e7eb"
      }
    }
  },
  plugins: []
};
