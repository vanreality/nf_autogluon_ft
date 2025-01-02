#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

process autogluon_train {
    input:
    tuple(
        val(name), 
        path(train), 
        path(validation), 
        path(test),
        val(usem),
        val(target),
        val(background)
    )

    output:
    path("${name}")                // Model output in work directory

    script:
    """
    mkdir -p ${name}
    python3 ${workflow.projectDir}/scripts/at.py \\
      --train ${train} \\
      --validation ${validation} \\
      --test ${test} \\
      --target ${target} \\
      --background ${background} \\
      --usem ${usem} \\
      --output ${name} &> ${name}/train.log
    """
}

workflow {
    Channel
        .fromPath(params.meta)
        .splitCsv(header: true, sep: '\t')
        .map { row -> 
            tuple(row.name, 
                  file(row.train), 
                  file(row.validation), 
                  file(row.test), 
                  row.usem.toBoolean(),
                  row.target,
                  row.background)
        }
        .set { ch_meta_rows }

    autogluon_train(ch_meta_rows)
}