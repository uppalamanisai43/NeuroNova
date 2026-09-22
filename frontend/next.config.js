/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    // Allow blob URLs for image preview
    domains: [],
  },
};

module.exports = nextConfig;
