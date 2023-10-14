## Bugs:
1. Missing error message when invalid ACBL player number entered into debug selectbox.
2. Find where score_NS_1 and score_EW_1 have creeped into df.
3. Don't let prompt create new columns. Check that dataframe results are all in results columns.
4. regex in commands.sql not working. SD_Prob_Take_[0-9]+ not excluded.
5. What to do with (tournament) sections not containing user's acbl_number? concat all sections by rows? concat may be correct if other sections' results are included in matchpoint calculations.

## Prompt Notes
1. Does chat understand overtricks, justmade, undertricks? Or should result be used?
2. Document SD_[Score,ParScore,Matchpoints,Pct][Max][Diff].
3. Any interesting stats to grab from https://stats.lovebridge.com/statistics/lb_02_107602 ?

## To Be Implemented
1. Is there some way that logo url can be using file:///assets/...?
2. Show list of acbl numbers, debug mode only, in landing page?
3. Create .json or .toml file for settings. Use settings to turn off any debug output.
4. Implement Checkbox for Debug Output in Advanced Settings. Use setting to turn off any debug output.
5. Setting to disable single dummy calculations (avoid delay)?
6. Don't perform string replacements on quoted prompts. Use as SQL.
7. ACBL Individuals
8. ACBL Teams
9. Other online bridge services.
10. Further reduce autosizing of columns. Dataframes doesn't look minimized although reportlab does.
11. Show predicted ranking for pairs. Requires NN to be trained on ranking. Might be more applicable to 7NT.info historical data.
13. Implement downloading of chat session or perhaps output as markdown.
17. Check if ACBL online games are in my DB.
20. Howell movements.
21. Write dataframe/SQL to in-memory SQL database? Perform queries on in-memory database instead of dataframe?
25. Show BBO bidding, both ACBL human and BBO robot.
26. Separate Release Notes from Project Notes?
27. Speed up single dummy. write to file or sql database?.
28. Highlight columns used in ORDER BY? Requires parsing of SQL statement.
29. Experiment with charts. Use ChartAI?
30. Flesh out README.md

## Prompt Issues
0. Prominantly show all boards where the scores are impossible given the contract. e.g. 2S for 120.
1. Check each prompt for missing statistics. Make list of minimum statistics.
2. Have expert review terminology used in prompts.
3. Underbid means par or sd score is higher than either a double or pass out.
4. Overbid means your contract result will be lower than parscore or sd score. although you might be able to make a higher contract in another strain.
5. Overbid prompt uses score < sd score. Should be making contract score < sd score. e.g. 2N < 129 is False.
6. Overbid prompt doubled contract looks like SD Contract Max is doubled too. Doesn't seem right.
7. Overbid prompt should use Declarer DD Score?
8. Implement list of best bids at any point in auction. It was done in 7NT.info?
9. actual, parscore, dd, sd, sd max -- need matrix of pcts and score diffs.
10. Show percentages loss/gain for in categories: gift, fix, luck, unlucky, ...
11. Develop prompt to show variance between actual scores, DD, and SD for same contract.
12. Distinguish underbids as contract mistakes if double dummy score < parscore and double dummy score < sd score.
13. Use emojis in prompts? 🥸
14. implement rankings. can this be done using a prompt or must webpage be scrapped?
15. Overall and directional rankings. Or can this be done using a prompt?
17. Historical Data: Calculate STDev of dealer DD, SD, Parscore, SD Max.
18. Show list of prompts in Help tab.
19. Create a single dataframe for DD, SD, SD Max, Par, and Actual instead of separate dataframes.
20. Does low SD Max correlate to bad bidding/contract?
21. What does low SD but high Pct say about a declarer? 
22. Show partnership declarer pct when showing declarer rankings.
23. Show par score pct and sd pcts for boards played by partnership.

## Column Issues
1. Rename any column names?
2. Drop useless columns.
3. What is the distinction between game_id and event_id? Are they currently used in the correct way?
4. What to do with BBO* columns?
5. Show historical data. Similar to 7NT.info.
6. Show pct lost/gained for each category: gift, fix, luck, unlucky, ...
7. What can be shown using LoTT? Can some type of matchpoint pct be shown?
8. What can be shown using Declarer_DD_SL_Diff (how much variance between DD and trump length?). Can some type of matchpoint pct be shown?

## Far Future
1. Ouput expert commentary about each board. Tag each board with flat, lucky, gift, fix, unlucky in preparation for commentary.
2. Rewrite using Polars? Need example of Polars using SQL.
3. Implement audio, avatar, video.
4. Analyze opening leads; nth longest, top of sequence, top of nothing, etc. Probably requires DD analysis of opening lead.
5. Show deal's cards, dd, sd, par using graphical output? Maybe not because ACBL link seems adequate enough.
6. Allow inputting of PBN file and its analysis.
7. Create command line interface which can read prompts (internal or external) and output report.
8. Implement play engine for analysis of play.
9. Implement bidding engine for analysis of bidding.

## Test
1. How many simultaneous users can be supported?
2. Verify mlBridgeLib.Lott is correct for all callers: Morty, 7NT, board_results.ipynb.

## Documentation Aids
1. https://markdown.land/markdown-cheat-sheet
2. https://www.color-hex.com/

## Related Projects Bugs
1. What's the diff between acbl_tournament_player_history and acbl_tournament_sessions?
2. Delete tournaments/players directory. Only use players and tournaments/sessions?
3. Looks like passed out hands are disappearing (row is dropped from results). Game ID 868296.

## Non-Deterministic Prompt responses:
1. CASE WHEN
2. Pct (ChatGPT 3.5 but happened once in 4)
3. Unique boards


## Experiments:
Experimental prompts:
		"Rankings": {
			"title": "Rankings",
			"help": "Show rankings by matchpoint percentage for pairs in your section and direction.",
			"prompts": [
				"Show distinct strat_rank_NS, final_standing_NS, strat_NS, section, strat_type_NS, player_name_n, player_name_s sort by strat_rank_NS ascending, final_standing_NS descending, strat_NS ascending, section ascending",
				"Show distinct strat_rank_EW, final_standing_EW, strat_EW, section, strat_type_EW, player_name_e, player_name_w sort by strat_rank_EW ascending, final_standing_EW descending, strat_EW ascending, section ascending"
			]
		},

## Enhancement Requests:

