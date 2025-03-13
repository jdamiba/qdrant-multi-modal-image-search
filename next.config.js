/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: ["localhost"],
    unoptimized: true,
  },
  // No need for rewrites as we're using Vercel's serverless functions
};

module.exports = nextConfig;
