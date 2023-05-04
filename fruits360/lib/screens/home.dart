import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite/tflite.dart';
import 'package:fruits360/utils/fruit_labels.dart';

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late File pickedImage;
  bool isImageLoaded = false;

  List _result = [];
  String _confidence = "";
  dynamic _name = "";
  String fruits = "";

  getImageFromGallery() async{
    var tempStore = await ImagePicker().pickImage(source: ImageSource.gallery);

    setState(() {
      pickedImage = File(tempStore!.path);
      isImageLoaded = true;
      applyModelOnImage(pickedImage);
    });
  }

  getImageFromCamera() async{
      var tempStore = await ImagePicker().pickImage(source: ImageSource.camera);

      setState(() {
        pickedImage = File(tempStore!.path);
        isImageLoaded = true;
        applyModelOnImage(pickedImage);
      });
    }

  resetImage() {
    setState(() {
      isImageLoaded = false;
      _result = [];
      _confidence = "";
      _name = "";
    });
  }

  loadCnnModel() async {
    var resultant = await Tflite.loadModel(
      model: "assets/cnn_model.tflite",
      labels: "assets/labels.txt",
    );

    print("Result after Loading Model : $resultant");
  }

  applyModelOnImage(File _file) async{
    var res = await Tflite.runModelOnImage(
      path: _file.path,
      numResults: 2,
      threshold: 0.5,
      imageMean: 127.5,
      imageStd: 127.5,
    );

    setState(() {
      _result = res!;
      print(_result);
      String str = _result[0]["label"];

      _name = fruitLabels[int.parse(str)];
      _confidence = (_result != null)? (_result[0]["confidence"]*100).toString().substring(0,2) + "%" : "";
    });
  }

  @override
  void initState() {
    // TODO: implement initState
    super.initState();
    loadCnnModel();
  }

  @override
  Widget build(BuildContext context) {
    Size pgSize = MediaQuery.of(context).size;
    return Scaffold(
      appBar: AppBar(
        toolbarHeight: pgSize.height*0.08,
        title: Text(widget.title,),
        backgroundColor: Color.fromARGB(255, 3, 36, 121),
      ),
      body: Container(
        alignment: Alignment.center,
        padding: const EdgeInsets.all(50),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: <Widget>[
            const Text(
              "Face Mask Detection",
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 50, color: Color.fromARGB(255, 15, 81, 157),),
            ),
            SizedBox(height: pgSize.height*0.1,),
            isImageLoaded?
            Container(
              height: pgSize.height * 0.3,
              width: pgSize.width * 0.5,
              alignment: Alignment.center,
              decoration: BoxDecoration(
                image: DecorationImage(
                  image: FileImage(File(pickedImage.path)),
                  fit: BoxFit.contain,
                ),
              ),
            )
            : Container(
              height: pgSize.height * 0.2,
              width: pgSize.width * 0.5,
              alignment: Alignment.center,
              child: ListTile(
                title: Image.asset("assets/coronavirus.png", width: pgSize.width*0.1,),
                subtitle: const Text(
                  "Upload an Image",
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: Colors.green,
                    fontSize: 18,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
            ),
            SizedBox(height: pgSize.height*0.05,),
            Text("Name : $_name\nConfidence : $_confidence"),
          ],
        ),
      ),
      floatingActionButton: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(width: pgSize.width*0.08,),
          FloatingActionButton(
            tooltip: 'Upload an Image',
            onPressed: () {
              getImageFromGallery();
            },
            child: const Icon(Icons.photo_library),
          ),
          const SizedBox(width: 25,),
          FloatingActionButton(
            tooltip: 'Upload an Image',
            onPressed: () {
              getImageFromCamera();
            },
            child: const Icon(Icons.camera_enhance),
          ),
          const SizedBox(width: 25,),
          FloatingActionButton(
            tooltip: 'Reset Image',
            onPressed: () {
              resetImage();
            },
            child: const Icon(Icons.replay_circle_filled),
          ),
        ],
      ),
    );
  }
}
