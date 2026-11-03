"""Graders and report scripts.

  bcp_judge          library: gpt-4.1 judge for BrowseComp-Plus and FRAMES (3 votes, majority, verdict cache)
  grade_bcp_snippet  script: grade a snippet REPL agent BrowseComp-Plus run directory
  grade_bcp_fulldoc  script: grade a document REPL agent BrowseComp-Plus run directory
  rejudge_bcp        script: re-judge stored answers of BrowseComp-Plus runs
  grade_frames       script: grade FRAMES run directories of either agent
  obliq_report       script: NDCG / recall of OBLIQ-Bench runs

bcp_judge and the scripts run code at import time; this __init__ imports nothing.
"""
