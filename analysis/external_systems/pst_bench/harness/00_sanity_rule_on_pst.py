"""Sanity check: run the OFFICIAL rule.find_paper_source_by_rule() evaluation loop on the released PST
labelled split (paper_source_trace_train_ans.json; the 2023 test file the function expects is not in
the public release). Only the file locations are re-pointed; the scoring logic is the official one.
Paper-reported rule MAP on the PST test set: 0.0616."""
import sys, os, json
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper-source-trace")
sys.path.insert(0, REPO); os.chdir(REPO)
import inspect, textwrap, rule, settings
src = inspect.getsource(rule.find_paper_source_by_rule)
src = (src.replace('utils.load_json(data_year_dir, "paper_source_trace_test.json")',
                   'utils.load_json(join(settings.DATA_TRACE_DIR, "PST"), "paper_source_trace_train_ans.json")')
          .replace('join(settings.DATA_TRACE_DIR, "paper-xml")', 'join(settings.DATA_TRACE_DIR, "PST", "paper-xml")')
          .replace('cur_pid + ".tei.xml"', 'cur_pid + ".xml"'))
exec(compile(textwrap.dedent(src), rule.__file__, "exec"), rule.__dict__)
rule.find_paper_source_by_rule()
