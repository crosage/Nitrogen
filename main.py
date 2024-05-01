import numpy as np
import matplotlib.pyplot as plt

def bsc_channel(input_bits, p):
    
    error_mask = np.random.rand(*input_bits.shape) < p
    output_bits = np.logical_xor(input_bits, error_mask).astype(int)
    return output_bits

def find_nearest(symbol,set):
    min_distance=888888888
    typical=None
    for s in set:
        s=np.array(list(s),dtype=int)
        distance=np.sum(np.abs(symbol-s))
        if distance<min_distance:
            typical=s
            min_distance=distance
    return typical

def joint_typical_decoding(receive_string, typical_set, codebook):
    """
    对接收到的码字解码

    先和典型值进行对比，找到最接近的典型值，然后典型值和标准值对比
    :param receive_string:
    :param typical_set:
    :param codebook:
    :return:返回解码的结果
    """
    typical=find_nearest(receive_string,typical_set)
    return find_nearest(typical,codebook)

def find_typical_set(codebook, n, q, epsilon):
    typical_set = []
    for seq in range(2**n):
        is_typical = False
        for codeword in codebook:
            seq_np = np.array(list(np.binary_repr(seq, width=n)), dtype=int)
            if q**np.sum(np.bitwise_xor(seq_np, codeword)) >= epsilon:
                is_typical = True
                break
        if is_typical:
            typical_set.append(np.binary_repr(seq, width=n))
    return typical_set

def generate_typical_set(n, epsilon):
    """
    主要解释: 典型值因为随机的缘故1的数量会更靠近0.5
    :param n:长度
    :param epsilon:典型集参数
    :return:典型集
    """
    typical_set = []
    M = 2 ** n
    for i in range(1, M):
        sequence = np.binary_repr(i, width=n)
        frequency = np.count_nonzero(np.array(list(sequence)) == '0') / n
        if 0.5-epsilon <= frequency <= 0.5 + epsilon:
            typical_set.append(sequence)

    return typical_set

def generate_random_codebook(n, R):
    """

    :param n:码字的长度
    :param R:码字的平均信息量
    :return:
    """
    codebook_size = int(2 ** (n * R))
    codebook = np.random.randint(2, size=(codebook_size, n))
    return codebook

def transport(counts, p, codebook, n, decoding_method, typical_set=None):
    error_avg = 0
    error_max = 0
    for i in range(counts):
        this_max = 0
        this_avg = 0
        for cnt in range(20):
            send_string = np.random.randint(len(codebook))
            receive_string = bsc_channel(codebook[send_string], p)
            if decoding_method == 'joint':
                decoded_msg = joint_typical_decoding(receive_string, typical_set, codebook)
            elif decoding_method == 'likelihood':
                decoded_msg = max_likehood_decoding(receive_string, codebook)
            else:
                raise ValueError("Invalid decoding method. Use 'joint' or 'likelihood'.")
            error_cnt = np.sum(decoded_msg != codebook[send_string]) / n
            this_max = max(this_max, error_cnt)
            this_avg += error_cnt
        this_avg /= 20
        error_avg += this_avg
        error_max += this_max
    error_avg /= counts
    error_max /= counts
    return error_avg, error_max

    

def draw(x, avg, max, x_label, title):
    plt.plot(x, avg, label='平均错误')
    plt.plot(x, max, label='最大错误')
    plt.xlabel(x_label)
    plt.ylabel('Error')
    plt.title(title)
    plt.legend()
    plt.show()

def max_likehood_decoding(receive_string, codebook):
    """
    直接使用最大似然译码
    :param receive_string:
    :param codebook:
    :return:返回解码的结果
    """
    return find_nearest(receive_string,codebook)

def join(parameter,parameter_range,n=10,epsilon=0.1,R=0.5,counts=50,p=0.1):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    results = {'parameter_values': [], 'error_avg': [], 'error_max': []}
    for value in parameter_range:
        results["parameter_values"].append(value)
        print(f"{parameter}:{value}")
        if parameter=='p':
            p=value
        elif parameter=='r':
            R=value
        elif parameter=='epsilon':
            epsilon=value
        elif parameter=='n':
            n=value     
        codebook=generate_random_codebook(n,R)
        typical=generate_typical_set(n,epsilon)
        a, m = transport(counts=counts, p=p,typical_set= typical,codebook= codebook, n=n,decoding_method="joint")
        results['error_avg'].append(a)
        results['error_max'].append(m)       
    draw(results['parameter_values'], results['error_avg'], results['error_max'], parameter, f"不同{parameter}下的平均错误概率和最大错误概率")

def draw_compare(x, join_avg, join_max,likelihood_avg,likelihood_max, x_label, title):
    plt.plot(x, join_avg, label='联合典型平均错误')
    plt.plot(x, join_max, label='联合典型最大错误')
    plt.plot(x, likelihood_avg, label='最大似然平均错误')
    plt.plot(x, likelihood_max, label='最大似然最大错误')
    plt.xlabel(x_label)
    plt.ylabel('Error')
    plt.title(title)
    plt.legend()
    plt.show()

def compete(parameter, parameter_range,n=10,epsilon=0.1,R=0.5,counts=50,p=0.1):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    results = {'parameter_values': [], 'join_error_avg': [], 'join_error_max': [], 'likelihood_error_avg': [], 'likelihood_error_max': []}
    for value in parameter_range:
        results['parameter_values'].append(value)
        print(value)
        if parameter == 'p':
            p = value
        elif parameter == 'n':
            n = value
        elif parameter == 'R':
            R = value
        codebook = generate_random_codebook(n, R)
        typical = generate_typical_set(n, epsilon)
        a, m = transport(counts=counts, p=p,typical_set= typical,codebook= codebook, n=n,decoding_method="joint")
        results['join_error_avg'].append(a)
        results['join_error_max'].append(m)
        a, m = transport(counts=counts, p=p,codebook= codebook, n=n,decoding_method="likelihood")
        results['likelihood_error_avg'].append(a)
        results['likelihood_error_max'].append(m)
    draw_compare(results['parameter_values'], results['join_error_avg'], results['join_error_max'],
                  results['likelihood_error_avg'], results['likelihood_error_max'], parameter,
                  f"不同{parameter}下的平均错误概率和最大错误概率")
    
if __name__=="__main__":

    plt.rcParams['font.sans-serif'] = ['SimHei']
    n=10
    r=0.5
    epsilon=0.1
    join_error_avg=[]
    join_error_max=[]
    thief_err_avg=[]
    thief_err_max=[]
    bad_avg=[]
    bad_max=[]
    P=[]
    counts=100
    codebook=generate_random_codebook(n,r)
    typical=generate_typical_set(n,epsilon)
    for p in np.arange(0,0.5,0.05):
        P.append(p)
        q=p+0.1
        a,m=transport(counts=counts, p=p,typical_set= typical,codebook= codebook, n=n,decoding_method="joint")
        join_error_avg.append(a)
        join_error_max.append(m)
        book=find_typical_set(codebook,n,p+0.05,0.05)
        aa,mm=transport(counts=counts, p=q,typical_set= typical,codebook= book, n=n,decoding_method="joint")
        thief_err_avg.append(aa)
        thief_err_max.append(mm)
        aaa,mmm=transport(counts=counts, p=p,typical_set= typical,codebook= codebook, n=n,decoding_method="joint")
        bad_avg.append(aaa)
        bad_max.append(mmm)
    plt.plot(P, join_error_avg, label='联合典型平均错误')
    plt.plot(P, join_error_max, label='联合典型最大错误')
    plt.plot(P, thief_err_avg, label='窃听平均错误')
    plt.plot(P, thief_err_max, label='窃听最大错误')
    plt.plot(P, bad_avg, label='窃听不做处理平均错误')
    plt.plot(P, bad_max, label='窃听不做处理最大错误')    
    plt.xlabel("p")
    plt.ylabel('Error')
    plt.title("窃听者（假设窃听者错误率高0.1）")
    plt.legend()
    plt.show()
    print(book)
