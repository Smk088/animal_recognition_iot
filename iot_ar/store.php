<?php
include("dbconnect.php");
extract($_REQUEST);
$rdate=date("d-m-Y");
$ch1=mktime(date('h')+5,date('i')+30,date('s'));
$rtime=date('h:i:s A',$ch1);

$mq=mysqli_query($connect,"select max(id) from animal_det");
$mr=mysqli_fetch_array($mq);
$id=$mr['max(id)']+1;

$ss=explode("/",$status);
$bcode=$ss[0];

//if($ss[1]=='a' || $ss[1]=='b')
//{
$qry=mysqli_query($connect,"insert into animal_det(id,details,sms_st,bcode,rdate,rtime) values($id,'$status','0','$bcode','$rdate','$rtime')");


mysqli_query($connect,"update animal_log set value2='$status' where bcode='$bcode'");


//mysqli_query($connect,"update borewell_log set value1='$ss[1]',page1=0 where bcode='$bcode'");
//}

/*if($qry)
{
//echo "success";
}
else
{
//echo "failed";
}*/

$q1=mysqli_query($connect,"select * from animal_log where bcode='$bcode'");
$r1=mysqli_fetch_array($q1);
//if($r1['value1']=='open')
//{
echo $r1['value1'];
//}
?>