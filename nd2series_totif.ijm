// nd2 series to tif
// 
// Liv Jensen
// Created 2019-05-22
// Modified 2019-05-31
// 
// This ImageJ macro imports nd2 files from within a given directory and outputs a folder of tif files 
// corresponding to each series within the nd2 file. 
// Runs on FIJI 2.0.0, has dependency on bio-formats

dir=getDirectory("Choose a Directory");
list = getFileList(dir);
for (i=0; i<list.length; i++) {
	if (endsWith(list[i], ".nd2")){
		impath=dir+list[i];
		outputDir= impath + "-output/";
		File.makeDirectory(outputDir);
		run("Bio-Formats", "open=[impath1] autoscale color_mode=Default open_all_series rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");
		run("Bio-Formats Macro Extensions");
		Ext.setId(impath);
		Ext.getSeriesCount(seriesCount);
		for (s=0; s<seriesCount; s++) {
			t = seriesCount-s;
			saveAs("Tiff", outputDir + "(series " + t + ").tif");
			close();
		}
	}
}