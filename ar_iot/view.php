<?php
include("dbconnect.php");
extract($_REQUEST);

//$status;
$q1=mysqli_query($connect,"select * from animal_log where bcode='$status' order by id desc");
$r1=mysqli_fetch_array($q1);

//echo 'z';
echo $r1['value2'];


?>