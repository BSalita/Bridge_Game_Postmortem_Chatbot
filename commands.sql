-- Experimenting with using text file of sql commands to make one-time changes to df.
-- CREATE OR REPLACE TABLE results_temp AS SELECT * FROM df; ALTER TABLE results_temp DROP COLUMN id;SELECT * from results_temp;
-- Doesn't look like regex is working to allow EXCLUDE of columns.
-- Using FROM df because registering of 'results' hasn't occurred yet. Won't occur until after this file is run.
SELECT * EXCLUDE (id, created_at, updated_at, board_id, Pair_Summary_ID_NS, Pair_Summary_ID_EW, Double_Dummy_NS, Double_Dummy_EW, SDProbs, SDScores) FROM df;