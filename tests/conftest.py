"""
Pytest configuration shared across the whole test suite.

Empty for now. A Hopsworks-backed version of this project needed this
file to stub out the `hopsworks` package before test collection --
push_to_hopsworks.py imported it unconditionally at module level, and
the real SDK was deliberately excluded from requirements-dev.txt for
being heavy, so importing that module at all (even just to patch one of
its functions in a test) required a fake stand-in.
`feature_pipeline/supabase_client.py` doesn't have that problem: the
`supabase` package is a plain HTTP client with no comparable weight, so
it's included directly in requirements-dev.txt (see Part 1 Step 8) and
tests import the real thing -- no stub needed. This file is kept around
as the conventional place to add shared fixtures later, not because
anything currently requires it.
"""