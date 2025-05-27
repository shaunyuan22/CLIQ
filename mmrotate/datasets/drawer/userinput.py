from multiprocessing import Manager
from mmcv.parallel import DataContainer as DC
import torch
import numpy as np

class UserinputDrawer:
    def __init__(self, *args, **kwargs):
        self.CLASSES = kwargs['classes']
        self.noc = kwargs['noc'] 
        self.max_user_input = kwargs['max_user_input'] 
        self.rand_sample = kwargs['rand_sample'] if rand_sample in kwargs else None
        manager = Manager()
        self.points = manager.dict()
        self.iter = 0
        

    def init_points_list(self, data):
        points_list, userinput_bbox_list = np.empty((0, ), int), np.empty((0, 2), float)
        bbox_count = min(len(data['gt_bboxes'].data), len(data['gt_labels'].data), self.max_user_input)
        if bbox_count > 0:
            num_points = np.random.randint(bbox_count)
            points_list = np.random.permutation(len(data['gt_bboxes'].data))[:num_points]
        return points_list, userinput_bbox_list

    def fill_insufficient_userinput(self, data, points_list, userinput_bbox_list):
        """Fill insufficient userinput for fixing shape."""
        insufficient_userinput = self.max_user_input - len(points_list)
        userinput_label_list = data['gt_labels'].data[points_list]
        userinput_label_list = np.append(userinput_label_list, np.ones((insufficient_userinput)) * -1)

        userinput_bbox_list = np.append(userinput_bbox_list, np.ones((insufficient_userinput * 2)) * -1).reshape(-1, 2)
        points_list = np.append(points_list, np.ones((insufficient_userinput)) * -1)
        return (points_list, userinput_bbox_list, userinput_label_list)
    
    def draw_points(self, data, points_list, img_shape, user_inputs, userinput_bbox_list=None):
        for point_idx in points_list:
            bbox, label = data['gt_bboxes'].data[point_idx], data['gt_labels'].data[point_idx]
            user_input, center_point = self.draw_point(bbox, img_shape)
            user_inputs[label].append(user_input)

            userinput_bbox_list = np.append(userinput_bbox_list, center_point)

        return user_inputs, userinput_bbox_list


    def prepare_user_input(self, data, idx):
        user_inputs, img_shape = self.prepare_empty_input(data)
        points_list, userinput_bbox_list = self.init_points_list(data)
        if points_list.size != 0:
            user_inputs, userinput_bbox_list = self.draw_points(data, points_list, img_shape, 
                                                                user_inputs, userinput_bbox_list)
            for i, user_input in enumerate(user_inputs):
                stack_user_input = np.stack(user_input)
                max_user_input = np.max(stack_user_input, axis=0)
                user_inputs[i] = max_user_input
            user_inputs = torch.tensor(np.array(user_inputs))
        else:
            user_inputs = torch.tensor(np.array(user_inputs)).squeeze(1)

        data['img'] = torch.cat((data['img'].data, user_inputs), 0)  \
                if isinstance(data['img'], DC) else [torch.cat((data['img'][0].data, user_inputs), 0)]

        data['gt_userinputs'] = self.fill_insufficient_userinput(data, points_list, userinput_bbox_list)
        return data
    
    def prepare_noc_point(self, data, user_inputs, idx, img_shape, userinput_bbox_list):
        bbox_count = min(len(data['gt_labels'].data), len(data['gt_bboxes'].data))

        # ensure least one simulate userinput
        if bbox_count > 0 and len(self.points[idx]) < bbox_count:
            bbox_idx = np.random.randint(bbox_count)
            while bbox_idx in self.points[idx] and len(self.points[idx]) < bbox_count:
                bbox_idx = np.random.randint(bbox_count)
            self.points[idx] += [bbox_idx]
        
        user_inputs, userinput_bbox_list = self.draw_points(data, self.points[idx], img_shape, user_inputs, userinput_bbox_list)
        return userinput_bbox_list, user_inputs
        
    def prepare_noc(self, data, idx):
        user_inputs, img_shape = self.prepare_empty_input(data)

        _, userinput_bbox_list = self.init_points_list(data)
        
        if idx not in self.points:
            self.points[idx] = []

        if self.iter != 0:
            userinput_bbox_list, user_inputs = self.prepare_noc_point(data, user_inputs, idx, img_shape, 
                                                                      userinput_bbox_list)

            points_list = self.points[idx]

            data['gt_userinputs'] = self.fill_insufficient_userinput(data, points_list, userinput_bbox_list)
        else:
            data['gt_userinputs'] = (np.ones((self.max_user_input)) * -1, 
                                     np.ones((self.max_user_input, 2)) * -1, 
                                     np.ones((self.max_user_input)) * -1)
        # user_inputs = torch.FloatTensor([np.amax(uh, axis=0) for uh in user_inputs])
        for i, user_input in enumerate(user_inputs):
            stack_user_input = np.stack(user_input)
            max_user_input = np.max(stack_user_input, axis=0)
            user_inputs[i] = max_user_input
        user_inputs = torch.tensor(np.array(user_inputs))

        data['img'] = torch.cat((data['img'].data, user_inputs), 0) \
                if isinstance(data['img'], DC) else [torch.cat((data['img'][0].data, user_inputs), 0)]
        return data

    def prepare_empty_input(self, data):
        img_metas = data['img_metas']
        img_shape = img_metas.data['pad_shape'] \
                        if isinstance(img_metas, DC) else img_metas[0].data['pad_shape']
        user_inputs = [[np.zeros(img_shape[:2], dtype=np.float64)] for _ in range(len(self.CLASSES))]
        return user_inputs, img_shape

    def sample_point_in_oriented_bbox(self, x, y, w, h, a):
        # Create rotation matrix
        cos_angle, sin_angle = torch.cos(a), torch.sin(a)
        rot_matrix = torch.tensor([[cos_angle, -sin_angle],
                                   [sin_angle, cos_angle]])
        # Scale the point to the width and height of the bounding box
        point[0] = point[0] * (w / 2)
        point[1] = point[1] * (h / 2)
        rotated_point = torch.matmul(rot_matrix, point)
        final_point = rotated_point + torch.tensor([x, y])
        return final_point[0], final_point[1]

    # def sample_point_in_oriented_bbox(self, x, y, w, h, a):
    #     # Create rotation matrix
    #     cos_angle, sin_angle = torch.cos(a), torch.sin(a)
    #     rot_matrix = torch.tensor([[cos_angle, -sin_angle],
    #                                [sin_angle, cos_angle]])
    #     # Generate random point from a normal distribution (standard normal distribution)
    #     min_w = min(0.9*w, 200)
    #     min_h = min(0.9*h, 200)
    #     random_w = torch.rand(1) * min_w - min_w / 2
    #     random_h = torch.rand(1) * min_h - min_h / 2
    #     point= torch.cat((random_w, random_h))
    #     rotated_point = torch.matmul(rot_matrix, point)
    #     final_point = rotated_point + torch.tensor([x, y])
    #     return final_point[0], final_point[1]

    def gaussian_2d(self, shape, centre, sigma=1.0):
        """Generate heatmap with single 2D gaussian."""
        xs = np.arange(0.5, shape[1] + 0.5, step=1.0, dtype=np.float64)
        ys = np.expand_dims(np.arange(0.5, shape[0] + 0.5, step=1.0, dtype=np.float64), -1)
        alpha = -0.5 / (sigma**2)
        heatmap = np.exp(alpha * ((xs - centre[0])**2 + (ys - centre[1])**2))
        return heatmap
    

    def draw_point(self, bbox, img_shape):
        if self.rand_sample is not None:
            cx, cy = self.sample_point_in_oriented_bbox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4])
            cx = torch.clamp(cx, min=0, max=img_shape[0] - 1)
            cy = torch.clamp(cy, min=0, max=img_shape[1] - 1)
            user_input = self.gaussian_2d(img_shape, [cx.numpy(), cy.numpy()])
            return user_input, [cx, cy]
        else:
            cx, cy = bbox[0], bbox[1]
            user_input = self.gaussian_2d(img_shape, [cx.numpy(), cy.numpy()])
            return user_input, [cx, cy]
            
