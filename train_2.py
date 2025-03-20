import tensorflow as tf
import numpy as np
import time
import datetime
import model
import get_data
import config
from sklearn.metrics import average_precision_score, roc_auc_score
from absl import flags, app
import os, sys
import timeit
FLAGS = flags.FLAGS

def MakeSummary(name, value):
    """Creates a tf.Summary proto with the given name and value."""
    summary = tf.Summary()
    val = summary.value.add()
    val.tag = str(name)
    val.simple_value = float(value)
    return summary

def Species_acc(pred, Y): #the smaller the better
    return np.mean(np.abs(pred - Y))

def Species_Dis(pred, Y): #the larger the better
    res = []
    for i in range(pred.shape[1]):
        try:
            auc = roc_auc_score(Y[:, i] ,pred[:, i])
            res.append(auc)
        except:
            res.append(1.0) #print("AUC nan", i, np.mean(Y[:, i]), np.mean(pred[:, i]))
        
    return np.mean(res)

def Species_Cali(pred, Y):
    res = []
    for j in range(pred.shape[1]):
        p = pred[:, j]
        y = Y[:, j]

        bin1 = np.zeros(10)
        bin2 = np.zeros(10)
        th = np.zeros(10)

        for k in range(10):
            th[k] = np.percentile(p, (k+1)*10)

        for i in range(p.shape[0]):
            for k in range(10):
                if (p[i] <= th[k]):
                    bin1[k] += p[i]
                    bin2[k] += y[i]
                    break

        diff = np.sum(np.abs(bin1 - bin2))
        #print(bin1)
        #print(bin2)
        res.append(diff)
    return np.mean(res)

def Species_Prec(pred, Y): #the smaller the better
    return np.mean(np.sqrt(pred * (1 - pred)))

def Richness_Acc(pred, Y): #the smaller the better
    return np.sqrt(np.mean((np.sum(pred, axis = 1)-np.sum(Y, axis = 1)) ** 2))

def Richness_Dis(pred, Y): #the larger the better
    return stats.spearmanr(np.sum(pred, axis = 1), np.sum(Y, axis = 1))[0]

def Richness_Cali(pred, Y): #the smaller the better
    samples = [] #100, n, sp
    for i in range(100):
        samples.append(np.random.binomial(1, pred))
    richness = np.sum(samples, axis = 2) #100, n
    gt_richness = np.sum(Y, axis = 1)

    res = []

    for i in range(pred.shape[0]):
        if (gt_richness[i] <= np.percentile(richness[:, i], 75) and gt_richness[i] >= np.percentile(richness[:, i], 25)):
            res.append(1)
        else:
            res.append(0)
    p = np.mean(res)
    return np.abs(p - 0.5)

def Richness_Prec(pred, Y): #the smaller the better
    samples = [] #100, n, sp
    for i in range(100):
        samples.append(np.random.binomial(1, pred))

    return np.mean(np.std(np.sum(samples, axis = 2), axis = 0))

def Beta_SOR(x, y):
    if (np.sum(x * y) == 0 and np.sum(x + y) == 0):
        return 0

    return 1 - 2 * np.sum(x * y)/np.maximum(np.sum(x + y), 1e-9)

def Beta_SIM(x, y):
    if (np.sum(x * y) == 0 and np.minimum(np.sum(x), np.sum(y)) == 0):
        return 0
    return 1 - np.sum(x * y)/np.maximum(np.minimum(np.sum(x), np.sum(y)), 1e-9)

def Beta_NES(x, y):
    return Beta_SOR(x, y) - Beta_SIM(x, y)

def get_dissim(pred, Y):
    samples = [] #100, n, sp
    for i in range(100):
        samples.append(np.random.binomial(1, pred))

    pairs = []
    N = 300
    for i in range(N):
        x = np.random.randint(pred.shape[0])
        y = np.random.randint(pred.shape[0])
        pairs.append([x, y])


    SOR = np.zeros((N, 100))
    SIM = np.zeros((N, 100))
    NES = np.zeros((N, 100))

    gt_SOR = []
    gt_SIM = []
    gt_NES = []
    for i in range(N):
        x, y = pairs[i]
        for j in range(100):
            SOR[i][j] = Beta_SOR(samples[j][x], samples[j][y])
            SIM[i][j] = Beta_SIM(samples[j][x], samples[j][y])
            NES[i][j] = Beta_NES(samples[j][x], samples[j][y])

        gt_SOR.append(Beta_SOR(Y[x], Y[y]))
        gt_SIM.append(Beta_SIM(Y[x], Y[y]))
        gt_NES.append(Beta_NES(Y[x], Y[y]))
    return SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES

def Community_Acc(pred, Y): #the smaller the better
    SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES = get_dissim(pred, Y)

    return np.sqrt(np.mean((np.mean(SOR, axis = 1) - gt_SOR)**2)),\
    np.sqrt(np.mean((np.mean(SIM, axis = 1) - gt_SIM)**2)),\
    np.sqrt(np.mean((np.mean(NES, axis = 1) - gt_NES)**2))

def Community_Dis(pred, Y): #the larger the better
    SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES = get_dissim(pred, Y)

    return stats.spearmanr(np.mean(SOR, axis = 1), gt_SOR)[0],\
    stats.spearmanr(np.mean(SIM, axis = 1), gt_SIM)[0],\
    stats.spearmanr(np.mean(NES, axis = 1), gt_NES)[0]

def Community_Cali(pred, Y):
    SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES = get_dissim(pred, Y)

    tmp1 = np.abs(np.mean(np.logical_and(np.less_equal(np.percentile(SOR, 25, axis = 1), gt_SOR),\
     np.greater_equal(np.percentile(SOR, 75, axis = 1),gt_SOR)).astype("float")) - 0.5)

    tmp2 = np.abs(np.mean(np.logical_and(np.less_equal(np.percentile(SIM, 25, axis = 1), gt_SIM),\
     np.greater_equal(np.percentile(SIM, 75, axis = 1),gt_SIM)).astype("float")) - 0.5)

    tmp3 = np.abs(np.mean(np.logical_and(np.less_equal(np.percentile(NES, 25, axis = 1), gt_NES),\
     np.greater_equal(np.percentile(NES, 75, axis = 1),gt_NES)).astype("float")) - 0.5)

    return tmp1, tmp2, tmp3

def Community_Prec(pred, Y): #the smaller the better
    SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES = get_dissim(pred, Y)

    return np.mean(np.std(SOR, axis = 1)), \
    np.mean(np.std(SIM, axis = 1)), \
    np.mean(np.std(NES, axis = 1))


def train_step(hg, input_X, input_Y, optimizer, step, writer):
    """Performs a training step, computes gradients, and logs metrics."""

    with tf.GradientTape() as tape:
        indiv_prob,  marginal_loss, l2_loss, total_loss = hg(is_training=True, n_features = n_features, n_classes = n_classes )

    # Compute gradients
    gradients = tape.gradient(total_loss, hg.trainable_variables)

    # Apply gradients
    optimizer.apply_gradients(zip(gradients, hg.trainable_variables))

    # Convert losses to scalars for easy debugging
    indiv_prob, nll_loss, marginal_loss, l2_loss, total_loss = indiv_prob.numpy(), nll_loss.numpy(), marginal_loss.numpy(), l2_loss.numpy(), total_loss.numpy()

    # Log metrics
    with writer.as_default():
        tf.summary.scalar("train/marginal_loss", marginal_loss, step=step)
        tf.summary.scalar("train/l2_loss", l2_loss, step=step)
        tf.summary.scalar("train/total_loss", total_loss, step=step)

    return indiv_prob, marginal_loss, l2_loss, total_loss


def validation_step(hg, DL, summary_writer, valid_idx, writer, epoch, metrics, metric_names):

    print('Validating...')

    valid_log = get_data.Log()


    batch_size = FLAGS.batch_size

    Ys = []
    preds = []
    for i in range((len(valid_idx)-1)//batch_size + 1):

        start = batch_size*i
        end = min(batch_size*(i+1), len(valid_idx))

        input_X = DL.get_X(valid_idx[start:end])
        input_Y = DL.get_Y(valid_idx[start:end])

        feed_dict={}
        feed_dict[hg.input_X]=input_X
        feed_dict[hg.input_Y]=input_Y
        feed_dict[hg.keep_prob]=1.0

        # Forward pass (no gradient calculation during validation)
        preds, nll_loss, marginal_loss, l2_loss, total_loss = hg(is_training=False, n_features=n_features, n_classes=n_classes)
        # Aggregate results
        all_nll_loss += nll_loss * len(batch_size)
        all_l2_loss += l2_loss * len(batch_size)
        all_total_loss += total_loss * len(batch_size)
        all_marginal_loss += marginal_loss * len(batch_size)   
        for ii in preds:
            print(ii.shape)
            preds.append(ii)
        for ii in input_Y:
            Ys.append(ii)
   
        # Compute average metrics
        mean_nll_loss = all_nll_loss / len(valid_idx)
        mean_l2_loss = all_l2_loss / len(valid_idx)
        mean_total_loss = all_total_loss / len(valid_idx)
        mean_marginal_loss = all_marginal_loss / len(valid_idx)

        # Compute average precision and AUC

        Ys = np.concatenate(Ys).flatten()
        preds = np.concatenate(preds).flatten()
        print(Ys)
        print(preds)
        ap = average_precision_score(Ys, preds)
    

    # Log results to TensorBoard
    with writer.as_default():

        tf.summary.scalar("validation/nll_loss", mean_nll_loss, step=step)
        tf.summary.scalar("validation/marginal_loss", mean_marginal_loss, step=step)
        tf.summary.scalar("validation/l2_loss", mean_l2_loss, step=step)
        tf.summary.scalar("validation/total_loss", mean_total_loss, step=step)

    return mean_nll_loss, preds, Ys

    

def main(_):
  
    st_time = time.time()
    print('Reading npy...')
    np.random.seed(19950420) # set the random seed of numpy
    DL = get_data.Data_loader(sys.argv[1], sys.argv[2], use_S = True)
   

    train_idx, test_idx, n_features, n_classes = DL.get_indices()
    #np.random.shuffle(train_idx)
    
    #M = len(train_idx)//10
    #valid_idx = train_idx[:M] 
    #train_idx = train_idx[M:]

    print("n_feature", n_features, "n_classes", n_classes)

    one_epoch_iter = len(train_idx) // FLAGS.batch_size # compute the number of iterations in each epoch

    print('reading completed')


    metrics = [Species_acc, Species_Dis, Species_Cali, Species_Prec, \
              Richness_Acc, Richness_Dis, Richness_Cali, Richness_Prec, 
              Community_Acc, Community_Dis, Community_Cali, Community_Prec]

    metric_names = ["Species_acc", "Species_Dis", "Species_Cali", "Species_Prec", \
                 "Richness_Acc", "Richness_Dis", "Richness_Cali", "Richness_Prec", \
                 "Community_Acc", "Community_Dis", "Community_Cali", "Community_Prec"]
    
    # GPU memory configuration
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    print('showing the parameters...\n')

    #parameterList = FLAGS.__dict__['__flags'].items()
    #parameterList = sorted(parameterList)

    # print all the hyper-parameters in the current training

    print('building network...')

    #building the model 
    hg = model.MODEL(is_training=True, n_features = n_features, n_classes = n_classes)

    global_step = tf.Variable(0, name='global_step', trainable=False)

  # Learning rate schedule
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True
    )

    # Optimizer setup
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    global_step = optimizer.iterations

    # Ensure checkpoint directory exists
    checkpoint_dir = FLAGS.model_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=hg)
    
    print ('building finished')
    
    # TensorBoard summary writer
    summary_writer = tf.summary.create_file_writer(FLAGS.summary_dir)

    best_loss = float('inf')
    current_step = global_step.numpy()  # Convert to integer
    best_loss = float('inf')
    best_epoch = 0
    drop_cnt = 0
    max_epoch = FLAGS.max_epoch
    for epoch in range(max_epoch):
        print(f'Epoch {epoch+1} starts!')
        
        # Shuffle training indices
        np.random.shuffle(train_idx)
        
        smooth_marginal_loss = 0.0
        smooth_l2_loss = 0.0
        smooth_total_loss = 0.0
        temp_Y = []
        temp_indiv_prob = []

        step = 0
        # Train for one epoch
        for i in range(len(train_idx) // FLAGS.batch_size):
            step += 1
            
            # Get batch
            start = i * FLAGS.batch_size
            end = (i + 1) * FLAGS.batch_size
            input_X = DL.get_X(train_idx[start:end])
            input_Y = DL.get_Y(train_idx[start:end])
            
            
            # Perform training step
            indiv_prob, nll_loss, marginal_loss, l2_loss, total_loss = train_step(hg, input_X, input_Y, n_features, n_classes, optimizer, current_step, summary_writer)

            # Update smooth losses
            smooth_marginal_loss += marginal_loss
            smooth_l2_loss += l2_loss
            smooth_total_loss += total_loss

            # Store labels and predictions for AP & AUC
            temp_Y.append(input_Y)
            temp_indiv_prob.append(indiv_prob)
            
            
            # Log progress every `check_freq` iterations
            if (i + 1) % FLAGS.check_freq == 0:
                mean_nll_loss = smooth_marginal_loss/FLAGS.check_freq
                mean_marginal_loss = smooth_marginal_loss/FLAGS.check_freq
                mean_l2_loss = smooth_l2_loss / FLAGS.check_freq
                mean_total_loss = smooth_total_loss / FLAGS.check_freq
                
                # Flatten and reshape predictions and labels
                temp_indiv_prob = np.reshape(np.array(temp_indiv_prob), (-1))
                temp_Y = np.reshape(np.array(temp_Y), (-1))


            # Compute AP & AUC
                ap = average_precision_score(temp_Y, temp_indiv_prob.reshape(-1, 1))

                try:
                    auc = roc_auc_score(temp_Y, temp_indiv_prob)
                except ValueError:
                    print('Warning: AUC computation failed due to label mismatch.')
                    auc = None 

                
                # Write to TensorBoard
                with summary_writer.as_default():
                    tf.summary.scalar('train/ap', ap, step=current_step)
                    if auc is not None:
                        tf.summary.scalar('train/auc', auc, step=current_step)
                    tf.summary.scalar('train/marginal_loss', mean_marginal_loss, step=current_step)
                    tf.summary.scalar('train/l2_loss', mean_l2_loss, step=current_step)
                    tf.summary.scalar('train/total_loss', mean_total_loss, step=current_step)
                
                time_str = datetime.datetime.now().isoformat()

                print ("train step: %s\tap=%.6f\tnll_loss=%.6f\tmarginal_loss=%.6f\tl2_loss=%.6f\ttotal_loss=%.6f" % (time_str, ap, nll_loss, marginal_loss, l2_loss, total_loss))
            
                # Reset accumulators
                temp_indiv_prob = []
                temp_Y = []
                smooth_marginal_loss = 0.0
                smooth_l2_loss = 0.0
                smooth_total_loss = 0.0
        save_epoch = 1
        # Validation
        if (epoch + 1) % save_epoch == 0:
            # Evaluate on test set
            Res = []
            for i in range(len(metrics)):
                f = metrics[i]
                name = metric_names[i]
                res = (name, f(preds, Ys))
                print(res)
                if (isinstance(res[1], tuple)):
                    for x in res[1]:
                        Res.append(x)
                else:
                    Res.append(res[1])
            
            test_loss, preds, Ys = validation_step(hg, DL, n_features, n_classes, summary_writer, test_idx, summary_writer, epoch, metrics, metric_names)
            
            
            print(f"Epoch {epoch+1} validation loss: {test_loss}")
            
            # Save if improved
            if test_loss < best_loss:
                print(f"New best loss: {test_loss} (previous: {best_loss})")
                best_loss = test_loss
                best_epoch = epoch
                best_res = [preds, Ys]
                checkpoint.save(file_prefix=checkpoint_prefix)

                drop_count = 0
            else:
                drop_count += 1
            
            # Early stopping
            if drop_count > 10:
                print("Early stopping triggered")
                break

    print('training completed !')
    print('the best loss on validation is '+str(best_loss))
    print('the best checkpoint is '+str(best_epoch))
    ed_time = timeit.default_timer()
    print("Running time:", ed_time - st_time)
    preds, Ys = best_res
    Res = []
    for i in range(len(metrics)):
        f = metrics[i]
        name = metric_names[i]
        res = (name, f(preds, Ys))
        print(res)
        if (isinstance(res[1], tuple)):
            for x in res[1]:
                Res.append(x)
        else:
            Res.append(res[1])

    np.save("results/%s_%s"%(sys.argv[1], sys.argv[2]), Res)
    





if __name__ == '__main__':
    app.run(main)
