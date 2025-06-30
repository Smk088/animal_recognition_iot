<?php
include("dbconnect.php");
extract($_REQUEST);



$q1=mysqli_query($connect,"select * from animal_log where bcode='$status'");
$r1=mysqli_fetch_array($q1);

echo $r1['value2'];


?>