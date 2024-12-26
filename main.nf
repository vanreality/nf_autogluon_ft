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
    path "${name}"  // Capture the output directory

    script:
    """
    python3 ${workflow.projectDir}/scripts/at.py \\
      --train ${train} \\
      --validation ${validation} \\
      --test ${test} \\
      --target ${target} \\
      --background ${background} \\
      --usem ${usem} \\
      --output ${name}
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
        .set { meta_rows_ch }

    autogluon_train(meta_rows_ch)
        .collect()
        .set { results }

    // Store process outputs to params.outdir
    results
        .map { dir -> dir.copyTo("${params.outdir}/${dir.name}") }
}