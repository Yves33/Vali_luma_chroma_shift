from PIL import Image,ImageShow
import numpy as np
import PyNvCodec as nvc
import os,sys

def make_img(buffer,w,h,format,merge=False):
    """
    utility function to build image from buffers. requires numpy, pillow. 
    
    Parameters
    ----------
    buffer:  Numpy array
    w, h :   Width and height of image, regardeless of buffer format
    format : Format of buffer. may be RGB, RGB_Planar, NV12, YUV420p
    merge :  Whether we should reconstitute rgb image or display the whole buffer (as grayscale image)
    """
    
    match format:
        case 'RGB':
            return Image.fromarray(buffer.reshape(h,w,3))
        case 'RGB_Planar':
            if not merge:
                return Image.fromarray(buffer.reshape(h*3,w))
            else:
                return Image.fromarray(buffer.reshape(3,h,w).transpose(2,1,0)).transpose(Image.TRANSPOSE)
        case 'NV12':
            if not merge:
                return Image.fromarray(buffer.reshape(h*3//2,w))
            else:
                y=Image.fromarray(buffer[:w*h].reshape(h,w))
                u=Image.fromarray(buffer[w*h:][::2].reshape(h//2,w//2).repeat(2, axis=0).repeat(2, axis=1))
                v=Image.fromarray(buffer[w*h+1:][::2].reshape(h//2,w//2).repeat(2, axis=0).repeat(2, axis=1))
                return Image.merge('YCbCr', (y,u,v))
        case 'YUV420p':
            if not merge:
                return Image.fromarray(buffer.reshape(h*3//2,w))
            else:
                y=Image.fromarray(buffer[:w*h].reshape(h,w))
                u=Image.fromarray(buffer[w*h:w*h*5//4].reshape(h//2,w//2).repeat(2, axis=0).repeat(2, axis=1))
                v=Image.fromarray(buffer[w*h*5//4:].reshape(h//2,w//2).repeat(2, axis=0).repeat(2, axis=1))
                return Image.merge('YCbCr', (y,u,v))

class TwoPassConverter:
    def __init__(self,fmt,w,h):
        self.to_yuv=nvc.PySurfaceConverter(fmt,nvc.PixelFormat.YUV420,gpu_id=0)
        self.to_rgb=nvc.PySurfaceConverter(nvc.PixelFormat.YUV420,nvc.PixelFormat.RGB,gpu_id=0)
        self.surf_yuv = nvc.Surface.Make(format=nvc.PixelFormat.YUV420, width=w,height=h,gpu_id=0)

    def Run(self,src,dst,cc_ctx):
        success,details=self.to_yuv.Run(src,self.surf_yuv,cc_ctx)
        if not success:
            return success, details
        success,details=self.to_rgb.Run(self.surf_yuv,dst,cc_ctx)
        return success, details

class OnePassConverter:
    def __init__(self,fmt,w,h):
        self.fmt, self.w, self.h=fmt, w, h
        self.to_rgb=nvc.PySurfaceConverter(fmt,nvc.PixelFormat.RGB,gpu_id=0)
        
    def Run(self,src,dst,cc_ctx):
        success,details=self.to_rgb.Run(src,dst,cc_ctx)
        return success, details

def check(success, info):
    if not success:
        print(sys._getframe(1).f_lineno)
        print(info)

def decode_vali(srcfile,nframes=20):
    '''
    decodes first nframes using vali. uses the new api (nvc.__version__>='3.2.0')
    convertion is performed either on gpu using vali converters or on cpu using pillow
    '''
    basename=srcfile.split('.')[-2]
    nvDec=nvc.PyDecoder(srcfile,{}, 0)
    rgbBuffer = np.zeros((nvDec.Width(),nvDec.Height(),3),np.uint8)
    nv12Buffer= np.zeros((nvDec.Width()*(3*nvDec.Height())//2),np.uint8)
    cSpace=nvDec.ColorSpace() if nvDec.ColorSpace()!=nvc.ColorSpace.UNSPEC else nvc.ColorSpace.BT_709
    cRange=nvDec.ColorRange() if nvDec.ColorRange()!=nvc.ColorRange.UDEF else nvc.ColorRange.MPEG
    cc_ctx = nvc.ColorspaceConversionContext(cSpace,cRange)
    cc_ctx = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_709,nvc.ColorRange.MPEG)
    print(f"#### {cSpace} {cRange} ####")
    if 0 and cSpace==nvc.ColorSpace.BT_709:
        nvCvt=TwoPassConverter(nvDec.Format(),nvDec.Width(),nvDec.Height())
    elif cSpace==nvc.ColorSpace.BT_601:
        nvCvt=OnePassConverter(nvDec.Format(),nvDec.Width(),nvDec.Height())

    nvDwn = nvc.PySurfaceDownloader(gpu_id=0)
    nvCvt=TwoPassConverter(nvDec.Format(),nvDec.Width(),nvDec.Height())
    for idx in range(nframes):
        rawSurface=nvc.Surface.Make(nvDec.Format(), nvDec.Width(),nvDec.Height(), 0)
        rgbSurface = nvc.Surface.Make(nvc.PixelFormat.RGB, nvDec.Width(),nvDec.Height(), 0)
        success, info = nvDec.DecodeSingleSurface(rawSurface)
        check(success,info)
        success = nvDwn.Run(rawSurface, nv12Buffer)
        check(success,info)
        success, info=nvCvt.Run(rawSurface,rgbSurface,cc_ctx)
        check(success,info)
        success = nvDwn.Run(rgbSurface, rgbBuffer)
        check(success,info)
        make_img(nv12Buffer,nvDec.Width(),nvDec.Height(),'NV12',merge=True).save(f"./{basename}/PIL_{idx+1:02}.jpg")
        make_img(rgbBuffer,nvDec.Width(),nvDec.Height(),'RGB',merge=True).save(f"./{basename}/VALI_{idx+1:02}.jpg")

def decode_vpf(srcfile, nframes=20):
    '''
    decodes first nframes using vpf.
    convertion is performed either on gpu using vali converters or on cpu using pillow
    '''
    basename=srcfile.split('.')[-2]
    ## vpf api is slightly different from vali
    nvDec=nvc.PyNvDecoder(srcfile, 0)
    rgbBuffer = np.zeros((nvDec.Width(),nvDec.Height(),3),np.uint8)
    nv12Buffer= np.zeros((nvDec.Width()*(3*nvDec.Height())//2),np.uint8)
    cSpace=nvDec.ColorSpace() if nvDec.ColorSpace()!=nvc.ColorSpace.UNSPEC else nvc.ColorSpace.BT_709
    cRange=nvDec.ColorRange() if nvDec.ColorRange()!=nvc.ColorRange.UDEF else nvc.ColorRange.MPEG
    cc_ctx = nvc.ColorspaceConversionContext(cSpace,cRange)
    nv12= (nvDec.Format()==nvc.PixelFormat.NV12)
    if nv12:
        nvYuv = nvc.PySurfaceConverter(nvDec.Width(),nvDec.Height(),nvDec.Format(), nvc.PixelFormat.YUV420, 0)
        nvRgb = nvc.PySurfaceConverter(nvDec.Width(),nvDec.Height(),nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB, 0)
    else:
        nvRgb = nvc.PySurfaceConverter(nvDec.Width(),nvDec.Height(),nvDec.Format(), nvc.PixelFormat.RGB, 0)
    nv12Dwn = nvc.PySurfaceDownloader(nvDec.Width(),nvDec.Height(),nvc.PixelFormat.NV12,0)
    rgbDwn = nvc.PySurfaceDownloader(nvDec.Width(),nvDec.Height(),nvc.PixelFormat.RGB,0)
    for idx in range(nframes):
        rawSurface = nvDec.DecodeSingleSurface()
        if nv12:
            yuvSurface = nvYuv.Execute(rawSurface,cc_ctx)
            rgbSurface = nvRgb.Execute(yuvSurface,cc_ctx)
        else:
            rgbSurface = nvRgb.Execute(rawSurface,cc_ctx)
        success = nv12Dwn.DownloadSingleSurface(rawSurface, nv12Buffer)
        success = rgbDwn.DownloadSingleSurface(rgbSurface, rgbBuffer)
        make_img(nv12Buffer,rawSurface.Width(),rawSurface.Height(),'NV12',merge=True).save(f"./{basename}/PIL_{idx+1:02}.jpg")
        make_img(rgbBuffer,rawSurface.Width(),rawSurface.Height(),'RGB',merge=True).save(f"./{basename}/VPF_{idx+1:02}.jpg")

def decode_ffmpeg(srcfile,nframes=20):
    basename=srcfile.split('.')[-2]
    os.system(f"ffmpeg -i {srcfile} -frames:v {nframes} ./{basename}/FFMPEG_%02d.jpg")

def decode_av(srcfile, nframes=20):
    basename=srcfile.split('.')[-2]
    try:
        import av
    except:
        print("AV not available. skipping test")
        return
    with av.open(srcfile) as container:
        stream = container.streams.video[0]
        for idx in range(nframes):
            frame=next(container.decode(stream))
            frame.to_image().save(f"./{basename}/AV_{idx+1:02}.jpg",quality=80)


if __name__=='__main__':
    srcfile="./hires.mp4"
    nframes=20
    basename=srcfile.split('.')[-2]
    os.makedirs(f"./{basename}/",exist_ok=True)
    if nvc.__maintainer__=="NVIDIA":
        decode_vpf(srcfile,nframes=nframes)
    elif nvc.__maintainer__=='Roman Arzumanyan':
        decode_vali(srcfile,nframes=nframes)
    #decode_ffmpeg(srcfile,nframes=nframes)
    #decode_av(srcfile,nframes=nframes)
    
