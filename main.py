import os
from flask import Flask, render_template, redirect, request, url_for, abort
from utils import random_name, upscale_image, get_psnr, get_ssim

if not os.path.isdir('static/original'):
    os.mkdir('static/original')

if not os.path.isdir('static/bicubic'):
    os.mkdir('static/bicubic')
    
if not os.path.isdir('static/compressed'):
    os.mkdir('static/compressed')
    
if not os.path.isdir('static/upscaled'):
    os.mkdir('static/upscaled')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        fl = request.files.get("file")
        if fl is None:
            return "No image uploaded!", 400

        nm = f"{random_name()}.jpg"
        original = f"original/{nm}"
        compressed = f"compressed/{nm}"
        upscaled = f"upscaled/{nm}"
        bicubic = f"bicubic/{nm}"

        fl.save("static/"+original)

        algo = request.form.get("algo")
        factor = request.form.get("factor")
        if algo is None or factor is None:
            return "Algorithm or Scaling Factor not specified!", 400

        w_original, h_original, w_compressed, h_compressed = upscale_image(f"static/{original}",
                                                                       f"static/{upscaled}",
                                                                       f"static/{compressed}",
                                                                       algo,
                                                                       factor,
                                                                       save_bicubic=True,
                                                                       bicubic_fname=f"static/{bicubic}")
        psnr_bicubic = f'{get_psnr(f"static/{original}", f"static/{bicubic}"):.4f}'
        psnr_upscaled = f'{get_psnr(f"static/{original}", f"static/{upscaled}"):.4f}'
        
        ssim_bicubic = f'{get_ssim(f"static/{original}", f"static/{bicubic}"):.4f}'
        ssim_upscaled = f'{get_ssim(f"static/{original}", f"static/{upscaled}"):.4f}'
        return redirect(url_for("result",
                        algo=algo,
                        factor=factor,
                        original=original,
                        upscaled=upscaled,
                        compressed=compressed,
                        bicubic=bicubic,
                        w_original=w_original,
                        h_original=h_original,
                        w_compressed=w_compressed,
                        h_compressed=h_compressed,
                        psnr_bicubic=psnr_bicubic,
                        psnr_upscaled=psnr_upscaled,
                        ssim_bicubic=ssim_bicubic,
                        ssim_upscaled=ssim_upscaled))
    return render_template("index.html")

@app.route("/result")
def result():
    algo = request.args.get('algo').upper()
    factor = request.args.get('factor')
    original = request.args.get('original')
    upscaled = request.args.get('upscaled')
    compressed = request.args.get('compressed')
    bicubic = request.args.get('bicubic')
    w_original = request.args.get('w_original')
    w_compressed = request.args.get('w_compressed')
    h_original = request.args.get('h_original')
    h_compressed = request.args.get('h_compressed')
    psnr_bicubic = request.args.get('psnr_bicubic')
    psnr_upscaled = request.args.get('psnr_upscaled')
    ssim_bicubic = request.args.get('ssim_bicubic')
    ssim_upscaled = request.args.get('ssim_upscaled')
    
    if not original or not upscaled or not compressed or not w_original or not w_compressed or not h_original or not h_compressed:
        abort(404)
        
    scale = {"2x" : 2, "4x" : 4}[factor]
    
    return render_template("result.html",
                            algo=algo,
                            scale=scale,
                            original=original,
                            upscaled=upscaled,
                            bicubic=bicubic,
                            compressed=compressed,
                            w_original=w_original,
                            h_original=h_original,
                            w_compressed=w_compressed,
                            h_compressed=h_compressed,
                            psnr_bicubic=psnr_bicubic,
                            psnr_upscaled=psnr_upscaled,
                            ssim_bicubic=ssim_bicubic,
                            ssim_upscaled=ssim_upscaled)

@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1,firefox=1'
    response.headers['Cache-Control'] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ == "__main__":
    app.run(debug=True)
