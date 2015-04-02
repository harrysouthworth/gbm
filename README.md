
# PLEASE START FOLLOWING gbm-developers/gbm:
https://github.com/gbm-developers/gbm

With the intention of ceasing to be such a bottleneck by having everything filtered through me, I've started the gbm-developers organization on GitHub.

Please switch to that version of gbm.

Thanks for your continuing contributions. They're much appreciated.

Harry

gbm: gradient boosted models
----------------------------

Originally written by Greg Ridgeway, added to by various authors,
currently maintained by Harry Southworth.  Development is discussed
--- somewhat --- at https://groups.google.com/forum/#!forum/gbm-dev .

Non-production releases (bug fixes, mostly) will be released via the GitHub
release workflow. To install from GitHub, first install `devtools` from CRAN:

```R
install.packages("devtools")
```

Then install `gbm` from GitHub:

```R
library("devtools")
install_github("gbm-developers/gbm")
```

Note that we are currently in the middle of a fairly major tidy up, so
you should worry a bit about this code here...