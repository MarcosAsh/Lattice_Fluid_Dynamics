import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Pin the React version so eslint-plugin-react skips its "detect" code path,
  // which calls context.getFilename() — removed in ESLint 10 and not yet fixed
  // in the eslint-plugin-react bundled by eslint-config-next.
  {
    settings: {
      react: { version: "19.2.7" },
    },
  },
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
  ]),
]);

export default eslintConfig;
