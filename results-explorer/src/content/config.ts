import { defineCollection, z } from "astro:content";

// Define base schemas
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

// Define overseer name as an enum
const overseerNameSchema = z.enum([
  "negligent_biased",
  "negligent_biased_judge",
  "negligent_biased_base",
]);

// Define common overseer parameters
const overseerSchema = z.object({
  name: overseerNameSchema,
  neglect_words: z.number().int().nonnegative(),
  bias_neglected_words: z.number().int().nonnegative(),
  bias_non_neglected_words: z.number().int().nonnegative(),
  bias_factor: z.number().nonnegative(),
  neglect_good_words: z.number().int().nonnegative(),
});

// Define the PreferenceSet schema
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

// Define collections
const negligentBiasedCollection = defineCollection({
  type: "data",
  schema: z.array(preferenceSetSchema),
});

const negligentBiasedBaseCollection = defineCollection({
  type: "data",
  schema: z.array(preferenceSetSchema),
});

const negligentBiasedBaseExtraCollection = defineCollection({
  type: "data",
  schema: z.array(preferenceSetSchema),
});

const negligentBiasedJudgeCollection = defineCollection({
  type: "data",
  schema: z.array(preferenceSetSchema),
});

// Export collections
export const collections = {
  "negligent-biased": negligentBiasedCollection,
  "negligent-biased-base": negligentBiasedBaseCollection,
  "negligent-biased-base-extra": negligentBiasedBaseExtraCollection,
  "negligent-biased-judge": negligentBiasedJudgeCollection,
};
