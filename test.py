def _load_CLEVRX(dataroot, name, img_id2val, label2ans, adaptive=True):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to
                retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(dataroot,
                                 'CLEVR-X/question_answers.json')
    image_data_path = os.path.join(dataroot,
                                   'CLEVR-X/image_data.json')
    ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
    cache_path = os.path.join(dataroot, 'cache', 'vg_%s%s_target.pkl' %
                              (name, '_adaptive' if adaptive else ''))

    if os.path.isfile(cache_path):
        entries = pickle.load(open(cache_path, 'rb'))
    else:
        entries = []
        ans2label = pickle.load(open(ans2label_path, 'rb'))
        vgq = json.load(open(question_path, 'r'))
        # 108,077 images
        _vgv = json.load(open(image_data_path, 'r'))
        vgv = {}
        for _v in _vgv:
            if _v['coco_id']:
                vgv[_v['image_id']] = _v['coco_id']
        # used image, used question, total question, out-of-split
        counts = [0, 0, 0, 0]
        for vg in vgq:
            coco_id = vgv.get(vg['id'], None)
            if coco_id is not None:
                counts[0] += 1
                img_idx = img_id2val.get(coco_id, None)
                if img_idx is None:
                    counts[3] += 1
                for q in vg['qas']:
                    counts[2] += 1
                    _answer = tools.compute_softscore.preprocess_answer(
                                q['answer'])
                    label = ans2label.get(_answer, None)
                    if label and img_idx:
                        counts[1] += 1
                        answer = {
                            'labels': [label],
                            'scores': [1.]}
                        entry = {
                            'question_id': q['qa_id'],
                            'image_id': coco_id,
                            'image': img_idx,
                            'question': q['question'],
                            'answer': answer}
                        if not COUNTING_ONLY \
                           or is_howmany(q['question'], answer, label2ans):
                            entries.append(entry)

        print('Loading CLEVR-X %s' % name)
        print('\tUsed CLEVR-X images: %d/%d (%.4f)' %
              (counts[0], len(_vgv), counts[0]/len(_vgv)))
        print('\tOut-of-split CLEVR-X images: %d/%d (%.4f)' %
              (counts[3], counts[0], counts[3]/counts[0]))
        print('\tUsed CLEVR-X questions: %d/%d (%.4f)' %
              (counts[1], counts[2], counts[1]/counts[2]))
        with open(cache_path, 'wb') as f:
            pickle.dump(entries, open(cache_path, 'wb'))

    return entries


class CLEVRXDataset(Dataset):
    def __init__(self, name, features, normalized_bb, bb,
                 spatial_adj_matrix, semantic_adj_matrix, dictionary,
                 relation_type, dataroot='data', adaptive=False,
                 pos_boxes=None, pos_emb_dim=64):
        super(CLEVR-XDataset, self).__init__()
        # do not use test split images!
        assert name in ['train', 'val']
        print('loading CLEVR-X data %s' % name)
        ans2label_path = os.path.join(dataroot, 'cache',
                                      'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache',
                                      'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.adaptive = adaptive

        self.img_id2idx = pickle.load(
                open(os.path.join(dataroot, 'imgids/%s%s_imgid2idx.pkl' %
                                  (name, '' if self.adaptive else '36')),
                     'rb'))
        self.bb = bb
        self.features = features
        self.normalized_bb = normalized_bb
        self.spatial_adj_matrix = spatial_adj_matrix
        self.semantic_adj_matrix = semantic_adj_matrix

        if self.adaptive:
            self.pos_boxes = pos_boxes

        self.entries = _load_CLEVR-X(dataroot, name, self.img_id2idx,
                                          self.label2ans,
                                          adaptive=self.adaptive)
        self.tokenize()
        self.tensorize()
        self.emb_dim = pos_emb_dim
        self.v_dim = self.features.size(1 if self.adaptive else 2)
        self.s_dim = self.normalized_bb.size(1 if self.adaptive else 2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * \
                          (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        raw_question = entry["question"]
        image_id = entry["image_id"]
        question = entry['q_token']
        question_id = entry['question_id']
        answer = entry['answer']
        if self.spatial_adj_matrix is not None:
            spatial_adj_matrix = self.spatial_adj_matrix[entry["image"]]
        else:
            spatial_adj_matrix = torch.zeros(1).double()
        if self.semantic_adj_matrix is not None:
            semantic_adj_matrix = self.semantic_adj_matrix[entry["image"]]
        else:
            semantic_adj_matrix = torch.zeros(1).double()
        if self.adaptive:
            features = self.features[
                self.pos_boxes[
                    entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            normalized_bb = self.normalized_bb[
                self.pos_boxes[
                    entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            bb = self.bb[self.pos_boxes[
                entry['image']][0]:self.pos_boxes[entry['image']][1], :]
        else:
            features = self.features[entry['image']]
            normalized_bb = self.normalized_bb[entry['image']]
            bb = self.bb[entry['image']]

        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)
        return features, normalized_bb, question, target, raw_question,\
            image_id, bb, spatial_adj_matrix, semantic_adj_matrix

    def __len__(self):
        return len(self.entries)