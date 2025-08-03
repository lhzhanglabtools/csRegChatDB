rm(list=ls())
library(BSgenome.Mmusculus.UCSC.mm10)
library(JASPAR2024)
library(chromVAR)
library(motifmatchr)
library(GenomicRanges)
motifs <- getJasparMotifs(species = "Mus musculus")
# write out the motifs used
motifs_name <- names(motifs)
motifs_new = strsplit(motifs_name,"_")
motifs_used = rep(NA,length(motifs_new),1)
for (i in 1:length(motifs_new)) {
  motifs_used[i] <- motifs_new[[i]][2]
}
write.table(motifs_used, file = "JASPAR_motifs.txt",sep = "\t", quote = FALSE)
# read used-peak data from BED file
peaks_bed = read.table(file = "peak_extract_mm10_filter.bed", sep = "\t")
colnames(peaks_bed) <- c("R.chrom","R.start","R.end")

source("D:/WHUer/notebooks/RegChatz/Code/Detect_Motif/get_peak_TF_links_misar.R")
memory.limit(size = 2048)
L_TF_record <- get_peak_TF_links(peaks_bed, species="Mus musculus", genome = BSgenome.Mmusculus.UCSC.mm10)
unik <- !duplicated(L_TF_record)
L_TF_record1 <- L_TF_record[unik,]
write.table(L_TF_record1,file = "MIASR_peaks_used_TFs_JASPAR2024.txt", sep = "\t")
