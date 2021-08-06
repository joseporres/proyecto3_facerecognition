var user_id;
$(function () {

   $('#blah').hide();

    $('#imageUploadForm').on('submit',(function(e) {
      e.preventDefault();
      var formData = new FormData(this);
      var option=$("#typeSearch").val();
      formData.append("typesearch", option);
      formData.append("k",$("#k").val());
      formData.append("n",$("#n").val());
      $( "body" ).addClass( "loader loader-default is-active" );
      $("#tcell").text("");
      $("#bred").text("");
      $("#bwhite").text("");
      $("#spots").text("");
      $("#result").text("");

      $.ajax({
          type:'POST',
          url: $(this).attr('action'),
          data:formData,
          cache:false,
          contentType: false,
          processData: false,
          success:function(data){
            // $.growl.error({ message: "The kitten is attacking!" });
            // $.growl.notice({ message: data.msg });
            // $.growl.warning({ message: "The kitten is ugly!" });
          // </script>
          $( "body" ).removeClass( "loader loader-default is-active" );
          
          var rpta = "";
          for (let step = 0; step < data.length; step++) {
            if(option == 1){
              var aux =data[step][1].split("/");
              rpta += (step+ 1).toString() + ". " + aux[aux.length-1].split(".")[0] + "\n";
              console.log(rpta);
            }
            else {
              var aux =data[step].split("/");
              rpta += (step+ 1).toString() + ". " + aux[aux.length-1].split(".")[0] + "\n";
              console.log(rpta);
            }
            
          }
          $("#ans").text(rpta);
          

          
          },
          error: function(data){
            $( "body" ).removeClass( "loader loader-default is-active" );
            // $.growl.error({ message: data.msg});
              console.log(data);
          }
      });
  }));

  $('.btn-file :file').on('change', function() {
    // console.log("entro1");
    var input = $(this),
        numFiles = input.get(0).files ? input.get(0).files.length : 1,
        label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
    input.trigger('fileselect', [numFiles, label]);

    var reader = new FileReader();
    reader.onload = function (e) {
        $('#blah')
            .attr('src', e.target.result)
            .width(500)
            .height(360)
            .show();
            
    };
    reader.readAsDataURL(input.get(0).files[0]);

  });
  
    $('.btn-file :file').on('fileselect', function(event, numFiles, label) {
      // console.log("entro2");
      var input = $(this).parents('.input-group').find(':text'),
          log = numFiles > 1 ? numFiles + ' files selected' : label;
      
      if( input.length ) {
          input.val(log);
      } else {
          if( log ) alert(log);
      }
  });

});



function filterItems(query, value, data_user) {
   if (query != "" || query != undefined) {
     return data_user.filter(function (el) {
       return el[value].toLowerCase().indexOf(query.toLowerCase()) > -1;
     });
   }
   return data_user;
 }


 function formatDate(date) {
   var hours = date.getHours();
   var minutes = date.getMinutes();
   var month = date.getMonth() + 1;
   var day = date.getDate();
   var ampm = hours >= 12 ? "pm" : "am";
   hours = hours % 12;
   hours = hours ? hours : 12; // the hour '0' should be '12'
   minutes = minutes < 10 ? "0" + minutes : minutes;
   var strTime = hours + ":" + minutes + " " + ampm;
   month = month < 10 ? "0" + month : month;
   day = day < 10 ? "0" + day : day;
   return day + "/" + month + "/" + date.getFullYear() + " " + strTime;
 }