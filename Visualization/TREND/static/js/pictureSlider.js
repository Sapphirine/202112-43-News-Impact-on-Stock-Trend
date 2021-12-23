class side{
    constructor(ct,imgs){
        let width = window.innerWidth;
        let height = window.innerHeight;
        this.width = width;
        this.height = height;
        this.currentPage = 0;
        this.pages = imgs.length;
        this.innerCt = document.createElement('div');
        this.innerCt.style.cssText = 'width:' + width * imgs.length + 'px;height:100%;padding:0px;margin:0px;transtion:teansform 1s ease';
        ct.appendChild(this.innerCt);
        let circleCt = document.createElement('div');
        circleCt.style.cssText = 'position:fixed;bottom:0px;width:100%;padding:10px 0';
        circleCt.setAttribute('align','center');
        ct.appendChild(circleCt);
        this.cicles = [];
        imgs.forEach((item,index) => {
            let ct1 = document.createElement('div');
            ct1.style.cssText = 'width:' + width +'px;height:' +height +'px;float:left;';
            ct1.setAttribute('align','center');
            let img = new Image();
            img.src = item;
            img.style.cssText = 'max-width:' + width + 'px;max-height:' + height +'px';
            img.onload = ()=>{
                img.style.marginTop = (height-img.height)/2 + 'px';
            }
            ct1.appendChild(img);
            this.innerCt.appendChild(ct1);
            let c = document.createElement('div');
            c.style.cssText = 'width:10px;height:10px;border-radius:5px;background-color:white;display:inline-block;margin-right:10px';
            this.cicles.push(c);
            circleCt.appendChild(c);
            c.addEventListener('click',()=>{
                this.sideTo(index);
            });
        });
        let css = 'position:absolute;top:50%;padding:0 10px;line-height:30px;background-color:#bbb;opacity:0.6;margin-top:-15px;font-size:18px;';
        let btnL = document.createElement('button');
        btnL.innerHTML = '<';
        btnL.style.cssText = css;
        let btnR = document.createElement('button');
        btnR.innerHTML = '>';
        btnR.style.cssText = css;
        btnL.style.left='0px';
        btnR.style.right = '0px';
        ct.appendChild(btnL);
        ct.appendChild(btnR);
        btnR.addEventListener('click',()=>{
            if(this.currentPage == this.pages -1){
                return;

            }
            this.sideTo(this.currentPage+1);
        });
        btnL.addEventListener('click',()=>{
            if(this.currentPage == 0){
                return;
            }
            this.sideTo(this.currentPage-1);
        });
        this.sideTo(0);
    }
    sideTo(num){
        this.cicles[this.currentPage].style.backgroundColor='white';
        this.cicles[num].style.backgroundColor='red';
        let left = - num *this.width;
        this.innerCt.style.transform = 'translate(' + left +'px,0px)';
        this.currentPage = num;
    }
}