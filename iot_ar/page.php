<?php
session_start();
include("dbconnect.php");
extract($_POST);
$msg="";
$uname=$_SESSION['uname'];

$q1=mysqli_query($connect,"select * from animal_log where uname='$uname'");
$r1=mysqli_fetch_array($q1);
$bc=$r1['bcode'];
//$mobile=$r1['mobile'];

if(isset($btn))
{
mysqli_query($connect,"delete from animal_det where bcode='$bc'");
?>
<script language="javascript">
window.location.href="page.php";
</script>

<?php
}




$q2=mysqli_query($connect,"select * from animal_log where uname='$uname'");
$r2=mysqli_fetch_array($q2);
if($r2['value1']!='')
{
mysqli_query($connect,"update animal_log set page1=1 where uname='$uname'");
}
if($r2['page1']=='1')
{
mysqli_query($connect,"update animal_log set page1=2 where uname='$uname'");
}
if($r2['page1']=='2')
{
mysqli_query($connect,"update animal_log set page1=3 where uname='$uname'");
}
if($r2['page1']=='3')
{
mysqli_query($connect,"update animal_log set value1='',value2='z',page1=0 where uname='$uname'");
}


?>
<html>
	<head>
	<meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <meta name="description" content="Start your development with Creative Design landing page.">
    <meta name="author" content="Devcrud">
    <title><?php include("title.php"); ?></title>

    <!-- font icons -->
    <link rel="stylesheet" href="assets/vendors/themify-icons/css/themify-icons.css">

    <!-- Bootstrap + Creative Design main styles -->
	<link rel="stylesheet" href="assets/css/creative-design.css">
		</head>
		<body>
<?php
					if($r2['value1']=='c')
					{
					?><h4 align="center">Buzzer Off</h4><?php
					}
					else if($r2['value1']=='d')
					{
					?><h4 align="center">Buzzer On</h4><?php
					}
					
					?>
<?php		
/////////////			

$ani_arr=array("a"=>"Elephant","b"=>"Monkey","c"=>"Lion","d"=>"Tiger","e"=>"Cheta","f"=>"Panda","g"=>"Fox","h"=>"Hyena","i"=>"Bison","j"=>"Leoprd","k"=>"Girafee","l"=>"Pig","m"=>"ostrich","n"=>"Dog");
					
					$v=$r1['value2'];
					if($v=="z")
					{
					
					
					}		
					else
					{
					?><h3 align="center" style="color:#FF6600">Animal: <?php echo  $ani_arr[$v]; ?></h3><?php
										
					}		
$qry=mysqli_query($connect,"select * from animal_det where bcode='$bc' order by id desc");
  $num=mysqli_num_rows($qry);
  
  if($num>0)
  {
  ?>
  <table width="100%" class="table" border="1" align="center" cellpadding="5">
    <tr>
      <th width="37" class="bg1" scope="row">Sno</th>
      <th width="292" class="bg1">Distance </th>
	  <th width="292" class="bg1">IR </th>
	  <th width="292" class="bg1">Direction </th>
      <th width="162" class="bg1">Date / Time </th>
    </tr>
	<?php
	$i=0;
	
	while($row=mysqli_fetch_array($qry))
	{
	$i++;
	$ss=explode("/",$row['details']);
	
	
	?>
    <tr>
      <td class="bg2" scope="row">
      <?php echo $i; ?></td>
      <td class="bg2"><?php 
	  echo $ss[1];
	  /*if($ss[1]=='a')
	  {
	  echo "Movement Detected Outer Lid Closing";
	  }
	  else if($ss[1]=='b')
	  {
	  echo "Movement Detected Inner Lid Closing Lifting Up";
	  }
	  else if($ss[1]=='c')
	  {
	  echo "Movement Cleared Lifting Down Inner Lid Opening";
	  }
	  else if($ss[1]=='d')
	  {
	  echo "Movement Cleared Lifting Down Outer Lid Opening";
	  }*/
	  
	   ?></td>
	   <td><?php echo $ss[2]; ?></td>
	   <td><?php echo $ss[3]; ?></td>
      <td class="bg2"><?php echo $row['rdate']." ".$row['rtime']; ?></td>
    </tr>
		
	<?php
		
		
	}
	?>
  </table>
 <p align="center"><input type="submit" name="btn" value="Delete All" /></p>
   <?php
  }
  else
  {
  echo '<div align="center">No Data Found!</div>';
  }
  ?>
  <script>
//Using setTimeout to execute a function after 5 seconds.
setTimeout(function () {
   //Redirect with JavaScript
   window.location.href= 'page.php';
}, 8000);
</script>

						</div>
					</div>
				</div>	
			</section>
			<!-- End contact-page Area -->
			
	
			

		</body>
	</html>



