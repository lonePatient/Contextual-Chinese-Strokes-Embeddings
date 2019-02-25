#encoding:Utf-8
class ProgressBar():
    def __init__(self,n_batch,
                 width=30):
        self.width = width
        self.n_batch = n_batch

    def batch_step(self,batch_idx,info,use_time):
        recv_per = int(100 * (batch_idx + 1) / self.n_batch)
        if recv_per >= 100:
            recv_per = 100
        show_bar = ('[%%-%ds]' % self.width) % (int(self.width * recv_per / 100) * ">")

        show_info = '\r[training] %d/%d %s -%.1fs/step '%(batch_idx+1,self.n_batch,show_bar,use_time)+\
                   "-".join([' %s: %.4f '%(key,value) for key,value in info.items()])

        print(show_info,end='')


