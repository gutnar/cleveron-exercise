# cleveron-exercise

Kasutasin ülesande lahendamiseks enamasti OpenCV võimalusi. Samuti leidsin palju infot aadressil https://chatbotslife.com/self-driving-cars-advanced-computer-vision-with-opencv-finding-lane-lines-488a411b2c3d, aga proovisin seal olevat vähe kasutada ja pigem eirata.

Kaustas OUTPUT on tulemused kõigi raw_images jaoks, kirjeldan siin tulemusi test1.jpg põhjal. Genereerisin ka videofaili road_processed.avi, kus on näitatud tulemused rakendatult videole road.mp4.

## Kaamera kalibreerimine

Uurisin, kuidas OpenCV-d kasutades leida pildil oleva ruudustiku põhjal kaamera kalibratsioonimaatriks. Leidsin, et kasutatakse funktsiooni nimega cv2.findChessboardCorners, mis võtab argumentideks pildi ja otsitavate sisemiste nurkade arvu ja võimaldab seeläbi seostada pildil pikslite ruumi tõelise ruumi koordinaatidega. Leitud koordinaatide põhjal saab kasutada funktsioone cv2.calibrateCamera, cv2.getOptimalNewCameraMatrix ja cv2.undistort kalibratsioonimaatriksi leidmiseks ja pildi korrekteerimiseks.

Proovisin kasutada kõiki antud viite pilti CALIBRATION kaustas, aga minu valitud sisemiste nurkade arvuga (9, 6) õnnestus seosed leida vaid kahe pildi puhul. Otsustasin, et praegu sellest piisab.

![](OUTPUT/calibration/1-corners2.jpg?raw=true "")

## Pildi korrekteerimine

Kasutades leitud kalibratsioonimaatriksit saan antud pildid korrekteerida. Näitan siin näiteid ühe pildi põhjal.

![](OUTPUT/test4/1-raw.jpg?raw=true "Korrekteeritud")
`cv2.undistort(img, self.mtx, self.dist, None, self.new_mtx)`
![](OUTPUT/test4/2-undistorted.jpg?raw=true "Korrekteeritud")

## Linnulennult vaade

Selle teisenduse tegemiseks leidsin aadressil https://nikolasent.github.io/opencv/2017/05/07/Bird's-Eye-View-Transformation.html hea selgituse. Põhimõtteliselt on vajalik tasapinnal ristküliku kujulise piirkonna kirjeldamine, mis originaalpildil on rööpküliku kujuline. Rööpküliku koordinaatide leidmiseks eraldasin videos kaadri, kus näis olevat sirge tee ning markeerisin punktid, mis moodustavad maapinnal ristküliku nagu näidatud pildil.
![](straight.jpg?raw=true "Sirge tee")
Kasutades funktsiooni cv2.warpPerspective saab projekteerida valitud piirkonna uuele pildile. Panin kõigepealt kohakuti valitud piirkonna ja uue pildi nurgad ning hakkasin siis uuel pildil sihtpunkte nihutama kuni tulemus tundus olevat piisavalt hea.

`cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]))`
![](OUTPUT/test4/3-birds-eye-view.jpg?raw=true "Linnulennult vaade")

## Teejoonte eraldamine

Teejoonte eraldamiseks alustasin värviruumi filtreerimisega ja mulle näis, et sellest piisab. HSV värvivahemike valimiseks alustasin ligikaudu kollasest ja valgest värvivahemikust ning optimiseerisin vahemikke videot läbi mängides.

`cv2.bitwise_and(grayscale_img, cv2.inRange(hsv, color_range[0], color_range[1]))`
![](OUTPUT/test4/4-color-mask.jpg?raw=true "Värviruumi filtreerimine")

## Teejoonte punktid ja joonte piirid

Alustuseks on teejoonte punktideks kõik värviruumi filtreerides leitud pikslid. Leidsin, et edaspidi kõiki punkte analüüsides on programm väga aeglane ning otsustasin valida igas kaadris väiksema alamvalimi punktidest (valides punkte ilma välja vahetamiseta). See tegi programmi oluliselt kiiremaks ja täpsust märgatavalt ei vähendanud.

Eeldasin, et pildi vasakule poole jäävad punktid kuuluvad vasaku joone alla ning paremale poole jäävad punktid paremale. Et välistada teatud määral müra, siis eemaldan punktid, kus x-telje suunaline dispersioon on suur, sest jooned võiksid olla kitsad.

Joonte kirjeldamiseks sobitan punkte numpy abil (np.polyfit) ruutfunktsiooniga (proovisin ka log sobitamist, mis ei andnud kuigi head tulemust). Pildil näitavad sinised täpid arvestatud punkte, punased täpid välistatud punkte ja rohelise piirkonna ääred sobitatud jooni.

![](OUTPUT/test4/6-fit.jpg?raw=true "Joonte sobitamine")

Video analüüsimisel tegin sujuvama tulemuse saamiseks nii, et sobitamisel arvestatakse mitme viimase kaadri jooksul leitud punkte. Mõtlesin, et võib-olla oleks tulemuslik ka auto liikumise kiirust ja suunda arvestades ennustada, kus eelmisel kaadril nähtud punktid järgmisel võiksid olla, aga peale mõtlemise sellega praegu ei tegelenud.

## Sõidurada, kurvi raadius ja sõiduki asukoht keskjoone suhtes

Tehes perspektiivi ja kaamera korrektsiooni tagasiteisendused, saan kuvada leitud raja esialgsel pildil. Tulemused üksikute piltide põhjal on enamasti halvemad kui video puhul, kus saab kasutada infot mitmest kaadrist.
![](OUTPUT/test4/7-final.jpg?raw=true "Tulemus")

Kurvi raadiuseks määrasin vasaku ja parema joone kui ruutfunktsioonide kõverused pildi alumises servas oleval kõrgusel. Vaatasin pildi järgi, et sellel kõrgusel vastab 1 meeter umbes 300 pikslile, et teisendada kõverus pikslitest meetriteks.

Raja keskjooneks määrasin samal kõrgusel kahe tuvastatud rajajoone vahelise keskmise punkti. Lisaks eeldasin, et kaamerapildi keskkoht langeb kokku sõiduki keskkohaga. Sõiduki asukoht keskjoone suhtes on siis nende suuruste vahe.
