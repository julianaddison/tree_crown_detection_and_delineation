# DESCRIPTION
A exploration of the approach discussed in SHAPE-BASED TREE CROWN DETECTION AND DELINEATION IN HIGH SPATIAL RESOLUTION AERIAL IMAGERY [Chen et. al.] (https://www.researchgate.net/publication/228808707_Shape-based_tree_crown_detection_and_delineation_in_high_spatial_resolution_aerial_imagery) is applied in splitting large contours for tree detection from aerial images.  

Simple image preprocessing is performed before contour detection. Image contrast adjustement is performed via a colour space transform (grayscale) and contrast limited adaptive histogram equalization. The watershed algorithm is used to find the contour of the processed image.

A prototype of the Chen et al's paper followed contour detection through nodularities. Large contours are identified as those with small nodularities. A mask of the large contour is created and a structure element i.e. normal tree size is slid across large contours to generate a similarity map. The peaks i.e.centers in the map are found. 

Contour delineation steps using geometry are applied to find split the large contours for overlapping peaks that are close together.

![alt text][logo]
[logo] https://github.com/julianaddison/tree_crown_detection_and_delineation/blob/master/results/se_sample.PNG "Concept diagram of the approach"

## INSTALLATION & USAGE
### TO RUN VIA DOCKER 

*TO BE INCLUDED IN FUTURE UPDATE*

### TO RUN LOCALLY
1. clone contents to a folder ./TreeCntsSplit

2. Launch Shell

3. Install required packages in requirement.txt
	
3. cd to filelocation

```bash
cd .\.\TreeCntsSplit\
```

4. Launch Jupyter Notebook
 
```bash
jupyter notebook
```

---

# LIMITATIONS
The approach is better suited for younger trees (<12M) where the region of overlap is lower, as seen in their paper OR where the planting regime is higher where mature trees do not have a high degree of overlap as in the sample image. Application may also be limited by the type of crown growth in different tree species e.g. Euclyptus vs Acacia trees.

The current implementation is a prototype and hence is not an efficient when used on a larger image. Modfications will need to be made. 

Contour delineation steps were not discussed in their paper either. The current implmentation utilizes geometry (go math!!!) when there could be better approach.
