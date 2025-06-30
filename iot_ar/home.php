<?php
session_start();
include("dbconnect.php");
extract($_REQUEST);
$uname=$_SESSION['uname'];
$rdate=date("d-m-Y");
$ch1=mktime(date('h')+5,date('i')+30,date('s'));
$rtime=date('h:i:s A',$ch1);

$q1=mysqli_query($connect,"select * from animal_log where uname='$uname'");
$r1=mysqli_fetch_array($q1);
$bc=$r1['bcode'];

?>
<html>
<head>
<!-- basic -->
<meta charset="utf-8">
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta name="viewport" content="width=device-width, initial-scale=1">
<!-- mobile metas -->
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="viewport" content="initial-scale=1, maximum-scale=1">
<!-- site metas -->
<title>AIRep: AI and IoT based Animal Recognition and Repelling System for Smart Farming</title>
<meta name="keywords" content="">
<meta name="description" content="">
<meta name="author" content="">	
<!-- bootstrap css -->
<link rel="stylesheet" type="text/css" href="css/bootstrap.min.css">
<!-- style css -->
<link rel="stylesheet" type="text/css" href="css/style.css">
<!-- Responsive-->
<link rel="stylesheet" href="css/responsive.css">
<!-- fevicon -->
<link rel="icon" href="images/fevicon.png" type="image/gif" />
<!-- Scrollbar Custom CSS -->
<link rel="stylesheet" href="css/jquery.mCustomScrollbar.min.css">
<!-- Tweaks for older IEs-->
<link rel="stylesheet" href="https://netdna.bootstrapcdn.com/font-awesome/4.0.3/css/font-awesome.css">
<!-- owl stylesheets --> 
<link rel="stylesheet" href="css/owl.carousel.min.css">
<link rel="stylesheet" href="css/owl.theme.default.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/fancybox/2.1.5/jquery.fancybox.min.css" media="screen">

</head>
<body>
	<!--header section start -->
	<div class="header_section">
		<div class="container">
			<div class="row">
				<div class="col-sm-2">
					<div class="logo"><a href="index.html"><img src="images/logo.png"></a></div>
				</div>
				<div class="col-sm-6">
					<div class="menu-area">
                    <nav class="navbar navbar-expand-lg ">
                        <!-- <a class="navbar-brand" href="#">Menu</a> -->
                        <button class="navbar-toggler collapsed" type="button" data-toggle="collapse" data-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                        <i class="fa fa-bars"></i>
                        </button>
                        <div class="collapse navbar-collapse" id="navbarSupportedContent">
                            <ul class="navbar-nav mr-auto">
                               <li class="nav-item active">
                                <a class="nav-link" href="userhome.php">Home<span class="sr-only">(current)</span></a> </li>
								
                               <li class="nav-item">
                                <a class="nav-link" href="logout.php">Logout</a></li>
                               
                            </ul>
                        </div>
                    </nav>
          </div>
				</div>
				<div class="col-sm-4">
            <ul class="top_button_section">
               <!--<li><a class="login-bt active" href="#">Login</a></li>
               <li><a class="login-bt" href="#">Register</a></li>-->
               <li class="search"><a href="#"><img src="images/search-icon.png" alt="#" /></a></li>
            </ul>
					</div>
			</div>

    <div class="row">
      <div class="banner_section layout_padding">
      <section>
         <div id="main_slider" class="section carousel slide banner-main" data-ride="carousel">
            <div class="carousel-inner">
               <div class="carousel-item active">
                    <div class="container">
                     <div class="row marginii">
                        <div class="col-md-5 col-sm-12">
                           <div class="carousel-sporrt_text ">
                             <h1 class="banner_taital">Animal Recognition</h1>
                    <p class="lorem_text">AIRep: AI and IoT based Animal Recognition and Repelling System for Smart Farming</p>
                    <div class="ads_bt"><a href="#">Ads Now</a></div>
                    <div class="contact_bt"><a href="#">Contact Us</a></div>
                           </div>
                        </div>
                        <div class="col-md-7 col-sm-12">
                           <div class="img-box">
                              <figure><img src="images/cow1.jpg" style="max-width: 90%; border:3px solid #009933; border-radius: 25px;"/></figure>
                           </div>
                        </div>
                    </div>
                  </div>
               </div>
               <div class="carousel-item">
                    <div class="container">
                     <div class="row marginii">
                        <div class="col-xl-6 col-lg-6 col-md-6 col-sm-12">
                           <div class="carousel-sporrt_text ">
                             <h1 class="banner_taital">Animal Recognition</h1>
                    <p class="lorem_text">AIRep: AI and IoT based Animal Recognition and Repelling System for Smart Farming</p>
                    <div class="ads_bt"><a href="#">Ads Now</a></div>
                    <div class="contact_bt"><a href="#">Contact Us</a></div>
                           </div>
                        </div>
                        <div class="col-xl-6 col-lg-6 col-md-6 col-sm-12">
                           <div class="img-box">
                              <figure><img src="images/banner-img1.png" style="max-width: 100%;"/></figure>
                           </div>
                        </div>
                    </div>
                  </div>
               </div>
               <div class="carousel-item">
                    <div class="container">
                     <div class="row marginii">
                        <div class="col-xl-6 col-lg-6 col-md-6 col-sm-12">
                           <div class="carousel-sporrt_text ">
                             <h1 class="banner_taital">Animal Recognition</h1>
                    <p class="lorem_text">AIRep: AI and IoT based Animal Recognition and Repelling System for Smart Farming</p>
                    <div class="ads_bt"><a href="#">Ads Now</a></div>
                    <div class="contact_bt"><a href="#">Contact Us</a></div>
                           </div>
                        </div>
                        <div class="col-xl-6 col-lg-6 col-md-6 col-sm-12">
                           <div class="img-box">
                              <figure><img src="images/banner-img1.png" style="max-width: 100%;"/></figure>
                           </div>
                        </div>
                    </div>
                  </div>
               </div>
            </div>
         </div>
      </section>
    </div>
    </div>

		</div>
        <!--header section end -->
        
		</div>
	</div>
    <!--banner section end -->
	<!--about section start -->
    <div class="about_section layout_padding">
    	<div class="container">
    		<div class="row">
    			<div class="col-md-12">
    				<div class="tablet">
					<div style="border:3px solid #009933; width:100%; height:500px; background-color:#FFFFFF; border-radius: 25px;">
					<h2 align="center">Animal Repellent</h2>
					
					<form name="form1" method="post">
					<div class="row">
						<div class="col-md-4">
						</div>
						<!--<div class="col-md-3">
						<input type="submit" class="btn btn-primary" name="b1" value="Lift Down">
						</div>-->
						<!--<div class="col-md-4">
						<input type="submit" class="btn btn-primary" name="b2" value="Buzzer Off">
						</div>-->
						
					</div>
					</form>
					<?php
					
						
					if(isset($b1))
					{
					mysqli_query($connect,"update animal_log set value1='c',page1=0 where uname='$uname'");
					
					$mq=mysqli_query($connect,"select max(id) from animal_det");
					$mr=mysqli_fetch_array($mq);
					$id=$mr['max(id)']+1;
					
					$vv=$uname."/c";
					
					$qry=mysqli_query($connect,"insert into animal_det(id,details,sms_st,bcode,rdate,rtime) values($id,'$vv','0','$bc','$rdate','$rtime')");


					}
					if(isset($b2))
					{
					mysqli_query($connect,"update animal_log set value1='open',page1=0 where uname='$uname'");
					$mq=mysqli_query($connect,"select max(id) from animal_det");
					$mr=mysqli_fetch_array($mq);
					$id=$mr['max(id)']+1;
					
					$vv=$uname."/d";
					
					$qry=mysqli_query($connect,"insert into animal_det(id,details,sms_st,bcode,rdate,rtime) values($id,'$vv','0','$bc','$rdate','$rtime')");
					
					}
					
					
					?>
					
                  <iframe src="page.php" width="100%" height="300" frameborder="0"></iframe>
						
					</div>
					
					</div>
    			</div>
    			
    		</div>
    	</div>
    </div>
    <div class="about_section_2 ">
    	
    </div>
	<!--about section end -->
	<!--client section start -->
    
	<!--client section end -->
	<!--footer section start -->
	<div class="footer_section layout_padding">
		<div class="container">
			<div class="row">
			    <div class="col-sm-3">
				    <div class="footer_contact">Farmer</div>
			    </div>
			    <div class="col-sm-3">
				    <div class="location_text"><img src="images/map-icon.png"><span style="padding-left: 10px;">Locations</span></div>
			    </div>
			    <div class="col-sm-3">
			    	<div class="location_text"><img src="images/call-icon.png"><span style="padding-left: 10px;">Mobile No.</span></div>
			    </div>
			    <div class="col-sm-3">
			    	<div class="location_text"><img src="images/mail-icon.png"><span style="padding-left: 10px;">E-mail</span></div>
			    </div>
		    </div>
		    <div class="enput_bt">
		    	
		    </div>
		    <div class="copyright_section">
		    	<p class="copyright_text"><a href="https://html.design"> Animal Recognition</p>
		    </div>
		</div>
	</div>
	<!--footer section end -->

      <!-- Javascript files-->
      <script src="js/jquery.min.js"></script>
      <script src="js/popper.min.js"></script>
      <script src="js/bootstrap.bundle.min.js"></script>
      <script src="js/jquery-3.0.0.min.js"></script>
      <script src="js/plugin.js"></script>
      <!-- sidebar -->
      <script src="js/jquery.mCustomScrollbar.concat.min.js"></script>
      <script src="js/custom.js"></script>
      <!-- javascript --> 
      <script src="js/owl.carousel.js"></script>
      <script src="https:cdnjs.cloudflare.com/ajax/libs/fancybox/2.1.5/jquery.fancybox.min.js"></script>
      <script>
         $(document).ready(function(){
         $(".fancybox").fancybox({
         openEffect: "none",
         closeEffect: "none"
         });
         </script>	


         <script>
      // This example adds a marker to indicate the position of Bondi Beach in Sydney,
      // Australia.
      function initMap() {
        var map = new google.maps.Map(document.getElementById('map'), {
          zoom: 11,
          center: {lat: 40.645037, lng: -73.880224},
          });

        var image = 'images/location_point.png';
          var beachMarker = new google.maps.Marker({
             position: {lat: 40.645037, lng: -73.880224},
             map: map,
             icon: image
          });
        }
        </script>
        <!-- google map js -->
          <script src="https://maps.googleapis.com/maps/api/js?key=AIzaSyA8eaHt9Dh5H57Zh0xVTqxVdBFCvFMqFjQ&callback=initMap"></script>
        <!-- end google map js -->
</body>
</html>
