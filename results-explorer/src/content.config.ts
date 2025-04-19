import { glob } from "astro/loaders";
import { defineCollection, z } from "astro:content";

const critiqueSchema = z.object({
  bad_word: z.string(),
  target_good_word: z.string(),
});

const oversightSchema = z.object({
  clue_critiques: z.object({
    clue: z.object({
      clue: z.string(),
      targets: z.array(z.string()),
    }),
    critiques: z.array(critiqueSchema),
  }),
  valid_targets: z.array(z.string()),
  expected_score: z.number().int(),
  ground_truth_score: z.number().int(),
  ground_truth_similarities: z.record(z.string(), z.number()),
  comparisons_performed: z.number().int(),
  deciding_critique: critiqueSchema.nullable(),
});

const overseerNameSchema = z.enum([
  "negligent_biased_base",
  "negligent_biased",
  "negligent_biased_judge",
]);

const overseerSchema = z.object({
  name: overseerNameSchema,
  neglect_words: z.number().int().nonnegative(),
  bias_neglected_words: z.number().int().nonnegative(),
  bias_non_neglected_words: z.number().int().nonnegative(),
  bias_factor: z.number().nonnegative(),
  neglect_good_words: z.number().int().nonnegative(),
});

const preferenceSetSchema = z.object({
  game: z.object({
    good_words: z.array(z.string()),
    bad_words: z.array(z.string()),
  }),
  overseer: overseerSchema,
  oversights: z.array(oversightSchema),
  adversarial_alpha: z.number().min(0).max(1),
  optimization_strength: z.number().int().positive().optional(),
});

const consultancy = defineCollection({
  loader: glob({
    pattern: "**/*.json",
    base: "./src/content/consultancy",
  }),
  schema: preferenceSetSchema,
});

const critiques = defineCollection({
  loader: glob({
    pattern: "**/*.json",
    base: "./src/content/critiques",
  }),
  schema: preferenceSetSchema,
});

const base = defineCollection({
  loader: glob({
    pattern: "**/*.json",
    base: "./src/content/base",
  }),
  schema: preferenceSetSchema,
});

export const collections = {
  consultancy,
  critiques,
  base,
};
