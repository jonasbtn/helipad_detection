

class FilterManager:
    
    """
    Class implementing 3 filtering methods: \n
        - Filter by IOU \n
        - Filter bounding contains inside another \n
        - Filter by score
    """
    @staticmethod
    def compute_interArea(boxA, boxB):
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
        # compute the area of a box
        boxArea = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        return boxArea

    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
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
        Filter the predicted bounding boxes overlapping of more than 50%
        Keep only the bounding box with the highest score

        if below 0.5, check if one box in contain inside the other
        like if the area of the intersection is above 90% of the area of one of the box
        and keep the one with the highest score

        `bboxes`:\n
        `class_ids`: \n
        `scores`: \n
        `threshold`: \n
        return \n
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
        Filter the box contained inside another
        Keep the boxes with the highest score or the biggest
        `bboxes`: \n
        `class_ids`: \n
        `scores`: \n
        `mode`: "highest" / "biggest" \n
        
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
        `bboxes`: \n
        `class_ids`: \n
        `scores`: \n
        `threshold`: \n
        returns \n
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
