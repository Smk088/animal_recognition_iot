<?php
include("dbconnect.php");
extract($_REQUEST);
$rdate=date("d-m-Y");
$ch1=mktime(date('h')+5,date('i')+30,date('s'));
$rtime=date('h:i:s A',$ch1);

$mq=mysqli_query($connect,"select max(id) from animal_det");
$mr=mysqli_fetch_array($mq);
$id=$mr['max(id)']+1;



//mysqli_query($connect,"update animal_det set details='$status',rdate='$rdate',rtime='$rtime' where bcode='$bc'");

mysqli_query($connect,"update animal_log set value2='$status' where bcode='$bc'");




//$qry=mysqli_query($connect,"insert into animal_det(id,details,rdate,rtime) values($id,'$status','$rdate','$rtime')");
if($qry)
{
echo "success";
}
else
{
echo "failed";
}

?>