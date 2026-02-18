configfile: "config/config.yaml"

include: "workflow/rules/common.smk"
include: "workflow/rules/dataset.smk"
include: "workflow/rules/tests.smk"
