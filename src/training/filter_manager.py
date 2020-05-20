

class FilterManager:
    
    """
    Class implementing 3 filtering methods: \n
        - Filter by IOU \n
        - Filter bounding contains inside another \n
        - Filter by score
    """
    @staticmethod
    def compute_interArea(boxA, boxB):
        """
        Compute the intersection area of two bounding boxes `boxA` and `boxB`
        """
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        return interArea

    @staticmethod
    def compute_area(box):
        """
        Computes the area of a bounding box `box`
        """
        boxArea = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        return boxArea

    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
        """
        Computes the IOU of two bounding boxes `boxA` and `boxB`
        """
        interArea = FilterManager.compute_interArea(boxA, boxB)

        boxAArea = FilterManager.compute_area(boxA)
        boxBArea = FilterManager.compute_area(boxB)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    @staticmethod
    def filter_by_iou(bboxes, class_ids, scores, threshold_iou=0.5, threshold_area=0.8):
        """
        Filter the predicted bounding boxes overlapping of more than `threshold_iou` (default:50%)\n
        Keep only the bounding box with the highest score\n

        If IOU is below 0.5, check if one box in contained inside another other \n
        if the area of the intersection is above `threshold_area` (default:80%) of the area of one of the bon
        Keep the one with the highest score \n

        `bboxes`: a list of bounding boxes\n
        `class_ids`: a list of class ids (0 or 1) corresponding to the bounding boxes\n
        `scores`: a list of confidence scores corresponding to the bounding boxes\n
        `threshold_iou`: the IOU threshold\n
        Return \n
        `new_bboxes`: filtered bounding boxes
        `new_class_ids`: filtered class id
        `new_scores`: filtered confidence score
        """
        added = [False]*len(bboxes)
        new_bboxes = []
        new_class_ids = []
        new_scores = []

        for i in range(len(bboxes)):
            boxA = bboxes[i]
            scoreA = scores[i]

            box_added = False

            for j in range(i+1, len(bboxes)):
                boxB = bboxes[j]
                scoreB = scores[j]

                interArea = FilterManager.compute_interArea(boxA, boxB)
                boxAArea = FilterManager.compute_area(boxA)
                boxBArea = FilterManager.compute_area(boxB)

                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the interesection area
                iou = interArea / float(boxAArea + boxBArea - interArea)

                # check threshold or if one is inside another
                if iou > threshold_iou or interArea > boxAArea*threshold_area or interArea > boxBArea*threshold_area:
                    # keep only the one with the biggest score
                    if scoreA > scoreB:
                        if not added[i]:
                            new_bboxes.append(boxA)
                            new_class_ids.append(class_ids[i])
                            new_scores.append(scores[i])
                            added[i] = True
                            added[j] = True
                            box_added = True
                    else:
                        if not added[j]:
                            new_bboxes.append(boxB)
                            new_class_ids.append(class_ids[j])
                            new_scores.append(scores[j])
                            added[i] = True
                            added[j] = True
                            box_added = True

            # if not found overlapping bboxes, add it
            if not box_added and not added[i]:
                new_bboxes.append(boxA)
                new_class_ids.append(class_ids[i])
                new_scores.append(scores[i])
                added[i] = True

        return new_bboxes, new_class_ids, new_scores

    @staticmethod
    def filter_contains(bboxes, class_ids, scores, mode="biggest"):
        """
        Filter the box contained inside another \n
        Keep the boxes with the highest score or the biggest \n
        
        `bboxes`: a list of bounding boxes\n
        `class_ids`: a list of class ids (0 or 1) corresponding to the bounding boxes\n
        `scores`: a list of confidence scores corresponding to the bounding boxes\n
        `mode`: "highest" / "biggest" \n
        
        Return \n
        `new_bboxes`: filtered bounding boxes
        `new_class_ids`: filtered class id
        `new_scores`: filtered confidence score
       
        """

        added = [False]*len(bboxes)
        new_bboxes = []
        new_class_ids = []
        new_scores = []

        for i in range(len(bboxes)):
            boxA = bboxes[i]
            scoreA = scores[i]

            box_added = False

            for j in range(i+1, len(bboxes)):
                boxB = bboxes[j]
                scoreB = scores[j]

                contains = -1
                # if boxA is contained inside boxB
                if boxA[0] > boxB[0] and boxA[1] > boxB[1] and boxA[2] < boxB[2] and boxA[3] < boxB[3]:
                    contains = 0
                # if boxB is contained inside boxA
                elif boxA[0] < boxB[0] and boxA[1] < boxB[1] and boxA[2] > boxB[2] and boxA[3] > boxB[3]:
                    contains = 1

                if contains != -1:
                    if mode == "highest":
                        # keep only the one with the biggest score
                        if scoreA > scoreB:
                            if not added[i]:
                                new_bboxes.append(boxA)
                                new_class_ids.append(class_ids[i])
                                new_scores.append(scores[i])
                                added[i] = True
                                added[j] = True
                                box_added = True
                        else:
                            if not added[j]:
                                new_bboxes.append(boxB)
                                new_class_ids.append(class_ids[j])
                                new_scores.append(scores[j])
                                added[i] = True
                                added[j] = True
                                box_added = True
                    elif mode == "biggest":
                        if contains == 0:
                            if not added[j]:
                                new_bboxes.append(boxB)
                                new_class_ids.append(class_ids[j])
                                new_scores.append(scores[j])
                                added[j] = True
                                added[i] = True
                                box_added = True
                        elif contains == 1:
                            if not added[i]:
                                new_bboxes.append(boxA)
                                new_class_ids.append(class_ids[i])
                                new_scores.append(scores[i])
                                added[i] = True
                                added[j] = True
                                box_added = True

            # if not found overlapping bboxes, add it
            if not box_added and not added[i]:
                new_bboxes.append(boxA)
                new_class_ids.append(class_ids[i])
                new_scores.append(scores[i])
                added[i] = True

        return new_bboxes, new_class_ids, new_scores

    @staticmethod
    def filter_by_scores(bboxes, class_ids, scores, threshold=0.995, threshold_validation=None, scores_validation=None):
        """
        Filter all the predicted bounding boxes below a certain threshold
        `bboxes`: a list of bounding boxes\n
        `class_ids`: a list of class ids (0 or 1) corresponding to the bounding boxes\n
        `scores`: a list of confidence scores corresponding to the bounding boxes\n
        `threshold`: float, score threshold \n
        `threshold_validation`: boolean to activate the score filtering using the second model validation of the bounding box\n
        `scores_validation`: float, score threshold \n
        
        Return \n
        `new_bboxes`: filtered bounding boxes
        `new_class_ids`: filtered class id
        `new_scores`: filtered confidence score
        """
        new_bboxes = []
        new_class_ids = []
        new_scores = []
        for i in range(len(scores)):
            if scores[i] < threshold:
                continue
            else:
                if threshold_validation:
                    if scores_validation[i] < threshold_validation:
                        continue
                new_bboxes.append(bboxes[i])
                new_class_ids.append(class_ids[i])
                new_scores.append(scores[i])
        return new_bboxes, new_class_ids, new_scores
