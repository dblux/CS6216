setwd("~/projects/phd/cs6216/")

# DATA --------------------------------------------------------------------
data_ps <- read.table('../diff_expr/data/breast_metastasis/GSE2034/processed/filtered_data.tsv',
                      header = T, row.names = 1)

# Imports probeset annotations
probeset_annot <- read.table("../info/microarray/HG-U133A/annot_entrez.tsv",
                             sep="\t", header=T, row.names=1, stringsAsFactors=F, strip.white = T)

fltr_annot <- probeset_annot[grepl("[0-9]_at", rownames(probeset_annot))
                             & !startsWith(rownames(probeset_annot), "A"), , drop=F]

# MAIN --------------------------------------------------------------------
# Returns entrez ID for all probe sets
entrez <- unname(sapply(rownames(data_ps), function(x) probeset_annot[x,]))
# # First entrez ID selected for ambiguous probe sets
# correction <- sub(" ///.*$", "", entrez[grepl("///", entrez)])
# # Entrez ID to be substituted
# entrez[grepl("///", entrez)] <- correction
# Indices of ambiguous probe sets and probe sets with no corresponding entrez ID to be deleted
list_del <- which(grepl("///", entrez) | entrez == "")

freq_gene <- table(entrez)
dup_genes <- names(freq_gene[freq_gene > 1])
for (i in dup_genes) {
  # Rows of dataframe with the same entrez ID
  same_rows <- data_ps[entrez == i,]
  # Assign indices as rownames
  rownames(same_rows) <- which(entrez == i)
  # Rows that do not have the maximum sum are deleted
  row_del <- as.integer(rownames(same_rows[-which.max(apply(same_rows,1,sum)),]))
  list_del <- c(list_del, row_del)
}
# Rows are deleted
data_genes <- data_ps[-list_del,]
fltr_entrez <- entrez[-list_del]
# Assigning entrez ID to df
rownames(data_genes) <- fltr_entrez

write.table(data_genes, "../diff_expr/data/breast_metastasis/GSE2034/processed/data_entrez1.tsv",
            sep = "\t", quote = F)
