## Acryonyms

DD is Double Dummy. DD is the definitive result of a particular board. DD is deterministic (always same).
SD is Single Dummy. SD is based on samplings of DD. The theory is that given enough samples of DD, an accurate probability distribution will emerge. SD has no knowledge of bidding.
Par Score is the score where neither direction can improve (aka Nash Equilibrium). It is based on the DD result, and vulnerability.
Strain is a suit or NT e.g. (C,D,H,S,N).

## Prompt Best Practices

Unfortunately the underlying AI is fickle. A prompt that works one minute might not work next minute. If you don't get proper results, try again later. If you still don't get proper results, try a different prompt.

### Recommendations

Let's deconstruct this example prompt: "Show board, contract, score, parscore, pct. Only boards I declared."

1. Begin with a list of desired column names. e.g. "Show board, contract, score, parscore, pct."
2. You can discover column names by familiarizing yourself with the Data, dtypes, or Schema tabs.
3. Use singular forms of words. e.g. "score", not "scores". "pct" not "pcts". Otherwise you might get both NS and EW columns.
4. Use a second sentence to provide only desired rows. e.g. "Show those boards I played." --or-- "Show those boards I declared." --or-- "Show those boards partner declared."
5. All column references should be in the list of desired column names. e.g. "board" is in the list of desired column names so it can be used in the second sentence("Show boards I played.").
6. Looks like chatgpt does better when directional column names (e.g. Score_EW) come before non-directional column names (e.g. SD Column Max). This is quite annecdotical, but it's worth a try if stuck.

