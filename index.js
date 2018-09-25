    let letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'T', 'U', 'W', 'X', 'Y', 'Z', ''];
    let dsize = new cv.Size(128, 64);
    let image = cv.imread(imgSource);       
    cv.cvtColor(image, image, cv.COLOR_BGR2GRAY, 0);
    cv.resize(image, image, dsize, cv.INTER_LINEAR);    
    image = image.data;
    image = tf.tensor2d(image, [128, 64]);
    image = tf.cast(image,'float32');
    image = tf.expandDims(image, -1);
    image = tf.expandDims(image, 0);           
    console.log("Model loading...");
    function predict(model, image){
       return tf.tidy(() => { 
         const netOutValue = model.predict(image)
         return netOutValue
       })
    }
    async function prob(){
      const model = await tf.loadModel("model4jszp/model.json");
      console.log("Model loaded.");  
      var netoutValue = predict(model, image);
      console.log(netoutValue.dataSync());
      var ret = [];
      var outBest = [];
      var outStr = "";
      outBest = tf.argMax(netoutValue.slice([0, 2, 0], [1, 30, 34]), 1).dataSync();
      console.log(outBest);
      for(let c of outBest){
        if(c<letters.length){
          outStr = outStr+letters[c];
        }
      }
      ret[ret.length] = outStr;
      console.log(ret);
    }
    prob();
