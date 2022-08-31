# Ship-Classification-using-triplet-CNN


Due to their all-weather, all-day, and high-resolution advantages, synthetic aperture radar (SAR) images have recently been used for ship classification
in marine surveillance. There are several satellites that have provided high-resolution SAR images since 2007, such as ASI’s COSMO-SkyMed, DLR’s TerraSAR-X, 
Japan’s ALOS-2, and China’s Gaofen-3, These high-resolution SAR images provide a resolution greater than 3 m that contain rich information about the targets, 
such as the geometry of ships, which makes discriminating different types of ships possible ['Cargo',' Dredging',' Passenger',' Tanker'].
The methods used for ship classification with SAR images mainly focus on feature selection and optimized classifier techniques. Currently,
commonly used features are geometric features, such as ship length, ratio of length to width, distribution of scattering centers, covariance coefficient,
contour features , and ship scale; and  scattering features, such as 2D comb features , local radar cross section (RCS) density,
permanent symmetric scatterers ,and polarimetric characteristics .
Based on the advantages of deep learning, convolutional neural networks (CNNs) are adapted in this paper. To further boost the feature discrimination, the task-specific
densely connected CNN is combined with the triplet networks to address the intraclass diversity and interclass similarity via the deep metric 
learning (DML) scheme. Compared to the normal CNNs, the triplet CNNs comprise three identical networks with each taking input an image of a triplet. One triplet is 
built by sampling an anchor image, a positive image of the same class to the anchor and a negative image of another class, from the training batch. 
Then, the sampled triplets are fed into the triplet CNNs to compute the triplet n   , by minimizing which the positive and anchor images are brought closer in the
learned feature space and the negative ones are pushed far apart from the anchors such that the intraclass compactness and interclass scatter are explicitly
encouraged, rendering the subsequent classification much simpler. 







******Please copy the files in dataset to the file ship_classiification******
