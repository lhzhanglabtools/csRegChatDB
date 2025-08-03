rm(list=ls())
setwd("/home/nas2/biod/zhencaiwei/RegChatz/Datasets/MISAR")
library(ArchR)
addArchRGenome("mm10")


# Creating a Custom ArchRGenome
library(BSgenome.Mmusculus.UCSC.mm10)
genomeAnnotation <- createGenomeAnnotation(genome = BSgenome.Mmusculus.UCSC.mm10)
genomeAnnotation
library(TxDb.Mmusculus.UCSC.mm10.knownGene)
library(org.Mm.eg.db)
geneAnnotation <- createGeneAnnotation(TxDb = TxDb.Mmusculus.UCSC.mm10.knownGene, OrgDb = org.Mm.eg.db)
genomeAnnotation


# Creating Arrow Files
ATACFiles <- "ArchR_inputs/mCortex_all_ATAC_fragment_filter_sorted.tsv.gz"
ATACNames <- c("ArchR_MISAR")
ArrowFiles <- createArrowFiles(
  inputFiles = ATACFiles,
  sampleNames = ATACNames,
  filterTSS = 4, #Dont set this too high because you can always increase later
  filterFrags = 1000, 
  addTileMat = TRUE,
  addGeneScoreMat = TRUE
)
ArrowFiles


# Creating An ArchRProject
projHeme1 <- ArchRProject(
  ArrowFiles = ArrowFiles, 
  outputDirectory = "ArchR_MISAR",
  copyArrows = TRUE #This is recommened so that if you modify the Arrow files you have an original copy for later usage.
)
projHeme1
paste0("Memory Size = ", round(object.size(projHeme1) / 10^6, 3), " MB")
getAvailableMatrices(projHeme1)


# Add Coverages
projHeme2 <- addGroupCoverages(ArchRProj = projHeme1, groupBy = "Sample")


# Calling Peaks w/ TileMatrix
projHeme3 <- addReproduciblePeakSet(
  ArchRProj = projHeme2,
  groupBy = "Sample",
  pathToMacs2 = "/home/zhencaiwei/anaconda3/bin/macs2"
)
peakset <- getPeakSet(projHeme3)
write.csv(peakset, file = "MISAR_peakset_ArchR_1.csv")


# Motif Enrichment in Differential Peaks
library(chromVARmotifs)
projHeme4 <- addMotifAnnotations(ArchRProj = projHeme3, motifSet = "cisbp", name = "Motif")
# Final output
Motif_Matches_In_Peaks <- readRDS('ArchR_MISAR/Annotations/Motif-Matches-In-Peaks.rds')
rownames <- rownames(Motif_Matches_In_Peaks)
colnames <- colnames(Motif_Matches_In_Peaks)
Motif_Matches_In_Peaks_data <- assay(Motif_Matches_In_Peaks)
Motif_Matches_In_Peaks_data <- Matrix::as.matrix(assay(Motif_Matches_In_Peaks))
write.csv(Motif_Matches_In_Peaks_data, file = "Motif_Matches_In_Peaks_data_1.csv")
