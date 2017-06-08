## facenet （TensorFlow version） ##

#### source git link ####

[https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet)

#### my cost function ####

	# f1(max) - f2(min)
	def triplet_loss(anchor, positive, negative, alpha):
	    """Calculate the triplet loss according to the FaceNet paper
	    Args:
	      anchor: the embeddings for the anchor images.
	      positive: the embeddings for the positive images.
	      negative: the embeddings for the negative images.
	    Returns:
	      the triplet loss according to the FaceNet paper as a float tensor.
	    """
	
	    with tf.variable_scope('triplet_loss'):
	
	        inner_thred = tf.constant(1, dtype=tf.float32)
	        outer_thred = tf.constant(2, dtype=tf.float32)
	
	        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
	        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
	        # positive
	        f1 = tf.maximum(pos_dist, inner_thred)
	        # negative
	        f2 = tf.minimum(neg_dist, outer_thred)
	        basic_loss = tf.add(tf.subtract(f1, inner_thred), tf.subtract(outer_thred, f2))
	
	        loss = tf.reduce_mean(basic_loss, 0)
	
	    return loss

#### my select triplet function ####

	# cqs add changed
	def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
	    """ Select the triplets for training
	    """
	    trip_idx = 0
	    emb_start_idx = 0
	    num_trips = 0
	    triplets = []
	
	    # VGG Face: Choosing good triplets is crucial and should strike a balance between
	    #  selecting informative (i.e. challenging) examples and swamping training with examples that
	    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
	    #  the image n at random, but only between the ones that violate the triplet loss margin. The
	    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
	    #  choosing the maximally violating example, as often done in structured output learning.
	
	    for i in range(people_per_batch):
	        nrof_images = int(nrof_images_per_class[i])
	        for j in range(1, nrof_images):
	            a_idx = emb_start_idx + j - 1
	            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
	            neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
	            # print("qingsong", (neg_dists_sqr[0:emb_start_idx] + neg_dists_sqr[emb_start_idx+nrof_images]).shape)
	            new_neg_dists_sqr = np.append(neg_dists_sqr[0:emb_start_idx], neg_dists_sqr[emb_start_idx+nrof_images:])
	            neg_dists_sqr_average = np.mean(new_neg_dists_sqr)
	            all_neg = np.where(neg_dists_sqr_average <= 2)[0]
	            nrof_random_negs = all_neg.shape[0]
	            for pair in range(j, nrof_images):  # For every possible positive pair.
	                p_idx = emb_start_idx + pair
	                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
	
	                if nrof_random_negs > 0 or pos_dist_sqr >= 1:
	                    if nrof_random_negs == 0:
	                        all_idxs = set(range(embeddings.shape[0]))
	                        pos_idxs = set(np.array(range(0, nrof_images)) + emb_start_idx)
	                        n_idx = list(all_idxs - pos_idxs)[0]
	                        # print("all_idxs", all_idxs, "pos_idxs", pos_idxs, "n_idx", n_idx)
	                    else:
	                        # rnd_idx = np.random.randint(nrof_random_negs)
	                        # n_idx = all_neg[rnd_idx]
	                        n_idx = np.nanargmin(new_neg_dists_sqr)
	                        if n_idx >= emb_start_idx:
	                            n_idx = n_idx + nrof_images
	                    print("nrof_random_negs percent:", nrof_random_negs / len(new_neg_dists_sqr), "neg_dists_sqr_average", neg_dists_sqr_average, "pos_dist_sqr", pos_dist_sqr)
	                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
	                    # print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
	                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
	                    trip_idx += 1
	
	                num_trips += 1
	
	        emb_start_idx += nrof_images
	
	    np.random.shuffle(triplets)
	    return triplets, num_trips, len(triplets)

#### cluster method 
    
code: src/aiphotoface
层次聚类
rankorder1
rankorder2
rankorder3
